import numpy as np
import matplotlib.pyplot as plt

def mi_funcion_sen(vmax=1, dc=0, ff=1, ph=0, nn=1000, fs=1000):
    tt = np.arange(0, nn) / fs
    velang = 2 * np.pi * ff
    xx = dc + vmax * np.sin(velang * tt + ph)
    return tt, xx

# ==================================================
# Valores de la senoidal
# ==================================================
N = 1000
fs = N
df = fs / N
ffs = N / 4      # tono en fs/4 = 250 Hz
dcs = 0
fas = 0
vnor = np.sqrt(2)

# ==================================================
# Valores del ADC y ruido
# ==================================================
B = 4
Vfs = 2
qq = 2 * Vfs / 2**B

k = 1
sigma = k * (qq**2) / 12
mu = 0

# ruido gaussiano incorrelado
U_n = np.random.normal(mu, np.sqrt(sigma), N)

# ==================================================
# Señales
# ==================================================
tt, xx = mi_funcion_sen(vmax=vnor, dc=dcs, ff=ffs, ph=fas, nn=N, fs=fs)

xxun = xx + U_n
xxq = np.round(xxun / qq) * qq

# ==================================================
# FFT
# ==================================================
frec = np.arange(N) * df

XX = np.fft.fft(xx)
XXun = np.fft.fft(xxun)
XXq = np.fft.fft(xxq)

# ==================================================
# Potencia bilateral y normalizada
# ==================================================
Pxx = (np.abs(XX)**2) / (N**2)
Pxxun = (np.abs(XXun)**2) / (N**2)
Pxxq = (np.abs(XXq)**2) / (N**2)

# ==================================================
# Espectro unilateral
# ==================================================
frec_half = frec[:N//2 + 1]

Pxx_uni = Pxx[:N//2 + 1].copy()
Pxxun_uni = Pxxun[:N//2 + 1].copy()
Pxxq_uni = Pxxq[:N//2 + 1].copy()

Pxx_uni[1:-1] *= 2
Pxxun_uni[1:-1] *= 2
Pxxq_uni[1:-1] *= 2

# ==================================================
# Conversión a dB
# ==================================================
Pxx_uni_db = 10 * np.log10(Pxx_uni + 1e-12)
Pxxun_uni_db = 10 * np.log10(Pxxun_uni + 1e-12)
Pxxq_uni_db = 10 * np.log10(Pxxq_uni + 1e-12)

# ==================================================
# Pisos teóricos (solo para referencia)
# ==================================================
piso_analogico_teo = 10 * np.log10(2 * sigma / N)
sigma_q = qq**2 / 12
piso_digital_teo = 10 * np.log10(2 * sigma_q / N)

# ==================================================
# Pisos prácticos
# ==================================================
bin_tono = int(ffs / df)

ruido_analogico_lin = np.delete(Pxxun_uni, [0, bin_tono])
ruido_digital_lin = np.delete(Pxxq_uni, [0, bin_tono])

piso_analogico_practico = 10 * np.log10(np.mean(ruido_analogico_lin))
piso_digital_practico = 10 * np.log10(np.mean(ruido_digital_lin))

print(f"Piso analógico teórico  = {piso_analogico_teo:.2f} dB")
print(f"Piso analógico práctico = {piso_analogico_practico:.2f} dB")
print()
print(f"Piso digital teórico    = {piso_digital_teo:.2f} dB")
print(f"Piso digital práctico   = {piso_digital_practico:.2f} dB")

# ==================================================
# Señales temporales
# ==================================================
fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

ax[0].plot(tt, xx)
ax[0].set_title('Señal analógica')
ax[0].grid(True)

ax[1].plot(tt, xxun)
ax[1].set_title('Señal con ruido')
ax[1].grid(True)

ax[2].step(tt, xxq, where='mid')
ax[2].set_title('Salida del ADC')
ax[2].set_xlabel('Tiempo [s]')
ax[2].grid(True)

plt.tight_layout()
plt.show()

# ==================================================
# Gráfico espectral
# ==================================================
plt.figure(figsize=(10, 6))

# plt.plot(frec_half, Pxx_uni_db, '--', label='Señal sin ruido')
# plt.plot(frec_half, Pxxun_uni_db, '--', label='Señal con ruido')
plt.plot(frec_half, Pxxq_uni_db, '--', label='Señal cuantizada')

# SOLO pisos prácticos
plt.axhline(piso_analogico_practico, color='green', linestyle=':',
            label=f'Piso analógico práctico = {piso_analogico_practico:.2f} dB')

plt.axhline(piso_digital_practico, color='purple', linestyle=':',
            label=f'Piso digital práctico = {piso_digital_practico:.2f} dB')

plt.xlabel('Frecuencia [Hz]')
plt.ylabel('Potencia [dB]')
plt.title('Espectro unilateral')
plt.ylim(-90, 10)
plt.xlim(0, fs/2)
plt.grid(True)
plt.legend()
plt.show()

# ==================================================
# Error de cuantización
# ==================================================
xxe = xxq - xxun

plt.figure(figsize=(8, 5))
plt.hist(xxe, bins=10, density=False, alpha=0.9, color='purple')
plt.axvline(qq/2, color='red', linestyle='--', label='+q/2')
plt.axvline(-qq/2, color='red', linestyle='--', label='-q/2')
plt.title('Histograma del error de cuantización')
plt.xlabel('Error (V)')
plt.ylabel('Frecuencia')
plt.legend()
plt.grid(alpha=0.3)
plt.show()