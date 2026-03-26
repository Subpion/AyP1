import numpy as np
import matplotlib.pyplot as plt

def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=1000, fs=1000):
    tt = np.arange(0, nn) / fs
    velang = 2 * np.pi * ff
    xx = dc + vmax * np.sin(velang * tt + ph)
    return tt, xx
from scipy import signal as sig
#%%
N = 1000
fs = 1000
ffs = 50
dcs = 0
fas = 0
vnor = np.sqrt(2)

tt, xx = mi_funcion_sen(vmax=vnor, dc=dcs, ff=ffs, ph=fas, nn=N, fs=fs)

SNR = 10
sigma = 10**(-SNR/10)
mu = 0
U_n = np.random.normal(mu, np.sqrt(sigma), N)

xxun = xx + U_n



B=3
Vfs= 3

qq=Vfs/2**B

xxq = np.round(xxun/qq)*qq
  
xxe= xxq-xxun


XX = np.fft.fft(xxun)
XXq = np.fft.fft(xxq)

f = np.fft.fftfreq(N, d=1/fs)

# Solo mitad positiva
idx = f >= 0
fpos = f[idx]

# Normalización correcta
XXmod = np.abs(XX[idx]) / N
XXqmod = np.abs(XXq[idx]) / N

# dB
XUdb = 20 * np.log10(XXmod + 1e-12)
XQdb = 20 * np.log10(XXqmod + 1e-12)

plt.figure()
plt.plot(fpos, XUdb, label='Señal con ruido')
plt.plot(fpos, XQdb, label='Señal cuantizada con ruido')
plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Magnitud [dB]')
plt.grid(True)
plt.legend()
plt.show()


















































