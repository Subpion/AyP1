import numpy as np
import matplotlib.pyplot as plt

# %%
N = 8
n = np.arange(N)

# Señal directamente en tiempo discreto
xx = 4 + 3 * np.sin((np.pi/2) * n)

# FFT
XX = np.fft.fft(xx)
XXabs = np.abs(XX)
XXang = np.angle(XX)

# Espectro de potencia
XXnorm = (XXabs**2) / N

# Eje de frecuencia en bins (más correcto en este caso)
k = np.arange(N)

# --- MODULO ---
plt.figure()
plt.stem(k, XXabs, linefmt='b-', markerfmt='bo', basefmt=" ")
plt.xlabel('k (bin)')
plt.ylabel('|X[k]|')
plt.title("FFT Módulo")
plt.grid(True)
plt.show()

# --- FASE ---
plt.figure()
plt.stem(k, XXang, linefmt='r-', markerfmt='ro', basefmt=" ")
plt.xlabel('k (bin)')
plt.ylabel('Fase [rad]')
plt.title('FFT Fase')
plt.grid(True)
plt.show()

# --- POTENCIA ---
plt.figure()
plt.stem(k, XXnorm, linefmt='g-', markerfmt='go', basefmt=" ")
plt.xlabel('k (bin)')
plt.ylabel('Potencia')
plt.title('FFT Espectro de potencia')
plt.grid(True)
plt.show()