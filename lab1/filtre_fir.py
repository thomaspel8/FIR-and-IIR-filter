import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
import scipy.signal as signal
from scipy.signal import firwin, lfilter, freqz

# Opening the audio file
fs, signal = wav.read("/Users/thomaspelletier/Downloads/speech_three-tones.wav")

# Converting the signal to a NumPy array
signal = np.array(signal)[:,0] # The audio signal is a stereo signal, so we decide to focus on the "left channel"
length = signal.shape[0] / fs
time = np.linspace(0., length, signal.shape[0])

# Calculating the spectrum of the input signal
fft = np.fft.fft(signal)
amplitude = np.abs(20*np.log(fft))

# Calculating the frequency of each component
freq = np.fft.fftfreq(len(signal), d=1/fs)

# Building the FIR filter
n = 1024  # Order of the filter
b = firwin(n, [600,1500], fs=fs, pass_zero = "bandpass", window = "hamming")

w, h = freqz(b)

# Applying the filter to the signal
signal_filtered = lfilter(b, 1, signal)

# Displaying the input signal
plt.plot(time, signal)
plt.xlabel("Temps")
plt.ylabel("Amplitude")
plt.title("Signal d'entrée")
plt.show()

# Displaying the spectrum of the input signal
plt.plot(freq, amplitude)
plt.xlabel("Fréquence")
plt.ylabel("Amplitude")
plt.title("Spectre en fréquences")
plt.show()

# Displaying the filter profile
plt.plot(w[0:500], 20*np.log(h)[0:500])
plt.xlabel("Fréquence")
plt.ylabel("Amplitude")
plt.title("Filtre passe bande")
plt.show()

# Displaying the filtered signal
plt.plot(time, signal_filtered)
plt.xlabel("Temps")
plt.ylabel("Amplitude")
plt.title("Signal de sortie")
plt.show()

# Writing the filtered signal to a WAV file
wav.write("/Users/thomaspelletier/Documents/Etudes/ECAM/ING5/Signal_Processing/labo_signal_processing/lab1/speech_three-tones_fir.wav"
, fs, np.int16(signal_filtered))