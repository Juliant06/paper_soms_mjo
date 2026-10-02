import scipy as sc 
import numpy as np 
import xarray as xr
import pandas as pd
from scipy.signal import detrend


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

    def remove_harmonics(self,):
        # remocion de los harmonicos
        if self.olr_wo_harms is None:
            olr_anti_sim = self.split_sim_antisim().copy()
            olr_remove_harmonic = np.zeros_like(olr_anti_sim)
            for i in range(olr_anti_sim.shape[2]):
                for j in range(olr_anti_sim.shape[3]):
                    data_a = pd.Series(
                        olr_anti_sim[0,:,i,j]
                    ).interpolate(method='linear').values
                    data_s = pd.Series(
                        olr_anti_sim[1,:,i,j]
                    ).interpolate(method='linear').values
                    olr_remove_harmonic[0,:,i,j] = self.remove_annual_harmonics(data_a)
                    olr_remove_harmonic[1,:,i,j] = self.remove_annual_harmonics(data_s)

            # Actualiza los datos en cache
            self.olr_wo_harms = olr_remove_harmonic
            return self.olr_wo_harms

        # envia el cache
        else:
            return self.olr_wo_harms

    def segments(self,):

        if self.segs_a is None and self.segs_s:
        
            seg_len, paso = 96, 36            # 96 días, traslape de 60
            N = self.olr.shape[0]
            olr_a = self.olr_wo_harms[0,:,:,:]
            olr_s = self.olr_wo_harms[1,:,:,:]
            inicios = range(0, N - seg_len + 1, paso)
            segs_a = np.stack([olr_a[i:i+seg_len] for i in inicios])  
            segs_s = np.stack([olr_s[i:i+seg_len] for i in inicios])

            # actualizar cache
            self.segs_a = segs_a
            self.segs_s = segs_s
            return self.segs_a, self.segs_s
        
        else:
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
        segs = np.staack([segs_a_tap, segs_s_tap])
         
        fft_lon = sc.fft.fft(segs, axis=4)
        fft_lon_time = sc.fft.ifft(fft_lon, axis=2)
        potencia = np.abs(fft_lon_time)**2
        esp = potencia.mean(axis=1).sum(axis=2) 

    
    








