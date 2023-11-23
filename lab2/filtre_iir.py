import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.signal import iirfilter, lfilter, freqz

# Ouvrir le fichier audio
fs, signal = wav.read("/Users/thomaspelletier/Downloads/speech_three-tones.wav")

# Convertir le signal en un tableau NumPy
signal = np.array(signal)[:,0] #le signal est un son stéréo, donc on décide de s'occuper du "côté gauche"
length = signal.shape[0] / fs
time = np.linspace(0., length, signal.shape[0])

# Calcul du spectre du signal filtré
fft = np.fft.fft(signal)
amplitude = np.abs(20*np.log(fft))

# Calculer la fréquence de chaque composante
freq = np.fft.fftfreq(len(signal), d=1/fs)

# Construction du filtre IIR
#n = 6  # Ordre du filtre
#b, a = iirfilter(n, [600,1500], fs=fs, btype = "bandpass", ftype = "butter")

#n = 6  # Ordre du filtre
#b, a = iirfilter(n, [600,1500], fs=fs, rs=60, rp=0.1, btype = "bandpass", ftype = "cheby1")

#n = 6  # Ordre du filtre
#b, a = iirfilter(n, [600,1500], fs=fs, rs=60, btype = "bandpass", ftype = "cheby2")

#n = 6  # Ordre du filtre
#b, a = iirfilter(n, [600,1500], fs=fs, btype = "bandpass", ftype = "bessel")

n = 6  # Ordre du filtre
b, a = iirfilter(n, [600,1500], fs=fs, rs=60, rp=0.1, btype = "bandpass", ftype = "ellip")

w, h = freqz(b, a)

#Application du filtre sur le signal
signal_filtered = lfilter(b, a, signal)

# Affichage du signal d'entré
plt.plot(time, signal)
plt.xlabel("Temps")
plt.ylabel("Amplitude")
plt.title("Signal d'entrée")
plt.show()

# Affichage du spectre en fréquences du signal
plt.plot(freq, amplitude)
plt.xlabel("Fréquence")
plt.ylabel("Amplitude")
plt.title("Spectre en fréquences")
plt.show()

# Affichage du profil du filtre
plt.plot(w / (2*np.pi), 20*np.log10(np.maximum(abs(h), 1e-5)))
plt.xlabel("Fréquence")
plt.ylabel("Amplitude")
plt.title("Filtre passe bande")
plt.show()

# Affichage du signal de sortie
plt.plot(time, signal_filtered)
plt.xlabel("Temps")
plt.ylabel("Amplitude")
plt.title("Signal de sortie")
plt.show()

# Écrire le signal filtré dans un fichier WAV
wav.write("/Users/thomaspelletier/Documents/Etudes/ECAM/ING5/Signal_Processing/labo_signal_processing/lab2/speech_three-tones_iir.wav", fs, np.int16(signal_filtered))