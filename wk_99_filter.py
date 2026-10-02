import numpy as np
import scipy as sc
import scipy.fft
import pandas as pd
import xarray as xr
from scipy.signal import detrend
from scipy.ndimage import convolve1d
from scipy.stats import chi2


# --------------------------------------------------------------------------
# Constantes y teoría de ondas ecuatoriales (aguas someras, plano beta)
# --------------------------------------------------------------------------
A_TIERRA, OMEGA, G = 6.371e6, 7.292e-5, 9.8
BETA = 2 * OMEGA / A_TIERRA


def _cpd(w):
    """rad/s -> ciclos/día."""
    return w * 86400 / (2 * np.pi)


def curva_dispersion(h, s, onda, n=1):
    """
    Frecuencia (cpd, > 0) de una onda ecuatorial para número de onda entero s.
    h: profundidad equivalente (m). onda: 'kelvin', 'mrg', 'eig0', 'er', 'wig', 'eig'.
    n: índice meridional (solo para 'er', 'wig', 'eig'). NaN donde la onda no existe.
    """
    c = np.sqrt(G * h)
    k = np.atleast_1d(np.asarray(s, float)) / A_TIERRA
    f = np.full(k.shape, np.nan)
    if onda == 'kelvin':                                   # w = c k
        m = k > 0
        f[m] = _cpd(c * k[m])
    elif onda in ('mrg', 'eig0'):                          # n = 0
        w = c * k / 2 + np.sqrt((c * k / 2) ** 2 + BETA * c)
        m = (k < 0) if onda == 'mrg' else (k > 0)
        f[m] = _cpd(w[m])
    else:                                                  # n >= 1: w^3 - (c²k² + (2n+1)βc) w - c²βk = 0
        for i, kk in enumerate(k):
            if kk == 0:
                continue
            r = np.roots([1, 0, -(c**2 * kk**2 + (2 * n + 1) * BETA * c), -c**2 * BETA * kk])
            r = np.sort(r[np.abs(r.imag) < 1e-12 * np.abs(r).max()].real)
            pos = r[r > 0]
            if len(pos) == 0:
                continue
            if onda == 'er' and kk < 0:
                f[i] = _cpd(pos[0])                        # raíz positiva menor
            elif onda == 'wig' and kk < 0:
                f[i] = _cpd(pos[-1])                       # raíz positiva mayor
            elif onda == 'eig' and kk > 0:
                f[i] = _cpd(pos[-1])
    return f


# Regiones de filtrado en el dominio k-w.
# MJO: del artículo (k = 1-5, periodos de 30 a 96 días, ambas componentes).
# Kelvin / ER / MRG: el texto solo da h = 8-90 m; los límites de k y de periodo son los
# habituales en la literatura posterior. VERIFICAR contra la Fig. 6 del artículo.
# comp: 0 = antisimétrica, 1 = simétrica, 'total' = A + S.
REGIONES = {
    'mjo':    dict(smin=1,   smax=5,  Tmin=30,  Tmax=96, onda=None, comp='total'),
    'kelvin': dict(smin=1,   smax=14, Tmin=2.5, Tmax=20, onda='kelvin', hmin=8, hmax=90, comp=1),
    'er':     dict(smin=-10, smax=-1, Tmin=9,   Tmax=72, onda='er', n=1, hmin=8, hmax=90, comp=1),
    'mrg':    dict(smin=-10, smax=-1, Tmin=3,   Tmax=10, onda='mrg', hmin=8, hmax=90, comp=0),
}


# --------------------------------------------------------------------------
# Clase principal
# --------------------------------------------------------------------------



class wk_99_filter: 

    def __init__(self,olr:xr.DataArray):
        self.olr = olr
        self.olr_wo_harms = None
        self.segs_a = None
        self.segs_s = None

    def split_sim_antisim(self,):
        olr_a = np.zeros(self.olr.shape)
        olr_s = np.zeros(self.olr.shape)
        lats = self.olr.lat.values

        # estimacion componentes simetricas y antisimetricas
        for idx,lat in enumerate(lats):
            array_lat = self.olr.sel(lat=lat).values
            array_n_lat = self.olr.sel(lat=-lat).values
            olr_a[:,idx,:] = (array_lat - array_n_lat)/2
            olr_s[:,idx,:] = (array_lat + array_n_lat)/2   
        olr_anti_sim = np.stack([olr_a,olr_s]) 
        return olr_anti_sim

    def remove_annual_harmonics(self,data, obs_per_day=1, n_harm=3):
        """
        data: array (tiempo, lat, lon), sin NaN (interpolar antes).
        Recorta a años enteros, elimina media + n_harm armónicos anuales.
        """
        obs_per_year = 365.25 * obs_per_day
        n_years = int(np.floor(data.shape[0] / obs_per_year))
        N = int(round(n_years * obs_per_year))     # longitud con años enteros
        data = data[:N]

        fhat = sc.fft.rfft(data, axis=0)
        fhat[0] = 0                                 # media
        for k in range(1, n_harm + 1):
            fhat[k * n_years] = 0                   # k ciclos/año = k*n_years ciclos totales
        return sc.fft.irfft(fhat, n=N, axis=0)

    def remove_harmonics(self):
        if self.olr_wo_harms is None:
            x = self.split_sim_antisim()
            N = int(round(int(np.floor(x.shape[1] / 365.25)) * 365.25))
            out = np.zeros((2, N, x.shape[2], x.shape[3]))
            for i in range(x.shape[2]):
                for j in range(x.shape[3]):
                    for c in (0, 1):
                        serie = pd.Series(x[c, :, i, j]).interpolate(method='linear').values
                        out[c, :, i, j] = self.remove_annual_harmonics(serie)
            self.olr_wo_harms = out
        return self.olr_wo_harms

    def segments(self):
        if self.segs_a is None:
            olr = self.remove_harmonics()
            seg_len, paso = 96, 36
            N = olr.shape[1]
            inicios = range(0, N - seg_len + 1, paso)
            self.segs_a = np.stack([olr[0, i:i+seg_len] for i in inicios])
            self.segs_s = np.stack([olr[1, i:i+seg_len] for i in inicios])
        return self.segs_a, self.segs_s

    def detrend_data(self,):
        
        segs_a, segs_s = self.segments()
        segs_a_detrend = detrend(segs_a,axis=1,type='linear')
        segs_s_detrend = detrend(segs_s,axis=1,type='linear')

        return segs_a_detrend, segs_s_detrend

    def split_cosine_bell(self, n=96, frac=0.1):
        w = np.ones(n)
        m = int(round(frac * n))                         # 10 puntos por extremo si n=96
        rampa = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
        w[:m] = rampa                                    # sube de 0 a ~1
        w[-m:] = rampa[::-1]                             # baja de ~1 a 0

        segs_a_detrend, segs_s_detrend = self.detrend_data()
        segs_a_tap = segs_a_detrend * w[None, :, None, None]
        segs_s_tap = segs_s_detrend * w[None, :, None, None]
        return segs_a_tap, segs_s_tap

    def wave_number_frequency(self,):

        segs_a_tap, segs_s_tap = self.split_cosine_bell()
        segs = np.stack([segs_a_tap, segs_s_tap])
         
        fft_lon = sc.fft.fft(segs, axis=4)
        fft_lon_time = sc.fft.ifft(fft_lon, axis=2)
        potencia = np.abs(fft_lon_time)**2
        esp = potencia.mean(axis=1).sum(axis=2) 

        return esp

    def kernel_121(self,n):
        """Kernel equivalente a n pasadas del filtro 1-2-1."""
        k, out = np.array([1., 2., 1.]) / 4, np.array([1.])
        for _ in range(n):
            out = np.convolve(out, k)
        return out

    def fondo(self, esp, nt, p_f=10, cortes=(0.1, 0.2), p_k=(10, 20, 40)):
        bg = esp.mean(axis=0)                                   # (se eliminó la línea que recalculaba esp)
        bg = convolve1d(bg, self.kernel_121(p_f), axis=0, mode='wrap')
        f_abs = np.abs(sc.fft.fftfreq(nt))
        bandas = [f_abs < cortes[0],
                (f_abs >= cortes[0]) & (f_abs < cortes[1]),
                f_abs >= cortes[1]]
        out = np.empty_like(bg)
        for m, n in zip(bandas, p_k):
            out[m] = convolve1d(bg[m], self.kernel_121(n), axis=1, mode='wrap')
        return out

    def cociente(self):
        esp = self.wave_number_frequency()
        nt, nk = esp.shape[1], esp.shape[2]
        bg = self.fondo(esp, nt)
        c = esp / bg
        f = sc.fft.fftfreq(nt); k = sc.fft.fftfreq(nk, 1 / nk)
        c = sc.fft.fftshift(c, axes=2); ks = sc.fft.fftshift(k)
        mf = f > 0; mk = np.abs(ks) <= 15
        return c[:, mf][:, :, mk], f[mf], ks[mk]



    ## Hecho por claude
    
    @staticmethod
    def umbral(n_dias, n_lat=13, p=0.95):
        """
        Umbral de significancia del cociente (chi2_dof / dof). Grados de libertad con el
        criterio del artículo: 2 * n_lat * (n_dias/96) * 0.57 (latitudes no independientes)
        * 0.5 (A y S por separado). Con 17.67 años reproduce ~1.1.
        """
        dof = 2 * n_lat * (n_dias / 96) * 0.57 * 0.5
        return chi2.ppf(p, dof) / dof, dof

    # ---------------- filtrado en el dominio k-w ----------------
    @staticmethod
    def mascara_region(f_fft, s_fft, reg):
        """Máscara booleana (nf, nk) en orden FFT para la región `reg` (ver REGIONES)."""
        nf, nk = len(f_fft), len(s_fft)
        mask = np.zeros((nf, nk), bool)
        lo_T, hi_T = 1 / reg['Tmax'], 1 / reg['Tmin']
        for j, s in enumerate(s_fft):
            if s < reg['smin'] or s > reg['smax']:
                continue
            lo, hi = lo_T, hi_T
            if reg['onda'] is not None:
                fa = curva_dispersion(reg['hmin'], [s], reg['onda'], reg.get('n', 1))[0]
                fb = curva_dispersion(reg['hmax'], [s], reg['onda'], reg.get('n', 1))[0]
                if np.isnan(fa) or np.isnan(fb):
                    continue
                lo, hi = max(lo, min(fa, fb)), min(hi, max(fa, fb))
            mask[:, j] = (f_fft > 0) & (f_fft >= lo - 1e-9) & (f_fft <= hi + 1e-9)
        # la señal es real: la máscara debe incluir también el espejo (-k, -f)
        espejo = mask[np.ix_((-np.arange(nf)) % nf, (-np.arange(nk)) % nk)]
        return mask | espejo

    def filtrar(self, data, reg, taper_dias=96):
        """
        Filtra `data` (T, lat, lon), con armónicos ya removidos, sobre el REGISTRO COMPLETO.
        Taper en los extremos de la serie (taper_dias por extremo; valor propio, el
        artículo no lo especifica), transformada directa, máscara k-w y transformada inversa.
        Devuelve (campo filtrado real, máscara).
        """
        T, _, nlon = data.shape
        w = np.ones(T)
        m = taper_dias
        rampa = 0.5 * (1 - np.cos(np.pi * np.arange(m) / m))
        w[:m] = rampa
        w[-m:] = rampa[::-1]
        X = sc.fft.ifft(sc.fft.fft(data * w[:, None, None], axis=2), axis=0)      # misma convención
        mask = self.mascara_region(sc.fft.fftfreq(T), sc.fft.fftfreq(nlon, 1 / nlon), reg)
        y = sc.fft.ifft(sc.fft.fft(X * mask[:, None, :], axis=0), axis=2).real    # inversa
        return y, mask

    def filtrar_onda(self, nombre):
        """Atajo: filtra la onda `nombre` de REGIONES usando la componente que le corresponde."""
        reg = REGIONES[nombre]
        olr = self.remove_harmonics()
        data = olr.sum(axis=0) if reg['comp'] == 'total' else olr[reg['comp']]
        return self.filtrar(data, reg)


# --------------------------------------------------------------------------
# Gráfica de verificación
# --------------------------------------------------------------------------
def graficar_cociente(c, f, k, umbral, salida=None):
    """Cociente potencia/fondo de A y S, sombreado donde supera el umbral, con curvas de h = 12, 25, 50 m."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    curvas = {0: [('mrg', 0), ('eig0', 0)], 1: [('kelvin', 0), ('er', 1)]}
    for i, titulo in enumerate(['Antisimétrica (A)', 'Simétrica (S)']):
        ax = axs[i]
        ax.contour(k, f, c[i], levels=np.arange(0, c[i].max() + 0.1, 0.1), colors='k', linewidths=0.3)
        ax.contourf(k, f, c[i], levels=[umbral, 1e9], colors=['0.75'])
        for onda, n in curvas[i]:
            for h, ls in [(12, ':'), (25, '-'), (50, '--')]:
                ax.plot(k, curva_dispersion(h, k, onda, n), 'r', ls=ls, lw=1)
        ax.axvline(0, color='k', ls='--', lw=0.5)
        ax.set_xlim(-15, 15)
        ax.set_ylim(0, 0.5)
        ax.set_title(titulo)
        ax.set_xlabel('Número de onda zonal (oeste < 0 < este)')
    axs[0].set_ylabel('Frecuencia (cpd)')
    if salida:
        fig.savefig(salida, dpi=120)
    return fig

def graficar_hovmoller(wk, inicio, fin, banda_lat=(-10, 2.5), ondas=('mjo', 'kelvin'), salida=None):
    """
    Diagramas tiempo-longitud (Hovmöller): anomalía de OLR total vs. campos filtrados por onda,
    promediados en la banda de latitud `banda_lat` (como las Figs. 8 y 9 del artículo).
    Tiempo hacia abajo. Azul = anomalía negativa de OLR (más convección).
    inicio, fin: fechas 'AAAA-MM-DD'. Evita los primeros y últimos ~100 días del registro
    (afectados por el taper del filtrado).
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
 
    olr = wk.remove_harmonics()                          # (2, N, lat, lon)
    N = olr.shape[1]
    tiempo = pd.to_datetime(wk.olr.time.values[:N])      # el registro se recortó a años enteros
    lat, lon = wk.olr.lat.values, wk.olr.lon.values
    mlat = (lat >= banda_lat[0]) & (lat <= banda_lat[1])
    mt = (tiempo >= pd.Timestamp(inicio)) & (tiempo <= pd.Timestamp(fin))
    if mt.sum() == 0:
        raise ValueError("Ningún dato entre `inicio` y `fin` (recuerda que el registro se recorta a años enteros).")
    fechas = mdates.date2num(tiempo[mt])
 
    titulos = {'mjo': 'MJO filtrada (k = 1-5, 30-96 d)', 'kelvin': 'Kelvin filtrada (S)',
               'er': 'ER n=1 filtrada (S)', 'mrg': 'MRG filtrada (A)'}
    campos = [('OLR total (anomalía, sin armónicos)', olr.sum(axis=0))]
    for nombre in ondas:
        y, _ = wk.filtrar_onda(nombre)
        campos.append((titulos.get(nombre, nombre), y))
 
    fig, axs = plt.subplots(1, len(campos), figsize=(4.6 * len(campos), 7), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, (titulo, campo) in zip(axs, campos):
        F = campo[mt][:, mlat, :].mean(axis=1)           # (tiempo, lon)
        v = np.nanpercentile(np.abs(F), 99)
        im = ax.pcolormesh(lon, fechas, F, cmap='RdBu_r', vmin=-v, vmax=v, shading='nearest')
        ax.set_title(titulo, fontsize=10)
        ax.set_xlabel('Longitud (°E)')
        ax.set_xlim(lon.min(), lon.max())
        fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.08, label='W m$^{-2}$')
    axs[0].yaxis_date()
    axs[0].yaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))
    axs[0].invert_yaxis()                                # tiempo hacia abajo
    axs[0].set_ylabel(f'Promedio {abs(banda_lat[0]):g}°S-{banda_lat[1]:g}°N')
    fig.tight_layout()
    if salida:
        fig.savefig(salida, dpi=120)
    return fig







