import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, ifft
from scipy import signal
import scipy
from sklearn.decomposition import NMF
import soundfile as sf
import os
# from IPython.display import Audio

def decompose(spectogram: np.ndarray, n_instruments: int, variant=''):
    n_components = n_instruments * 4

    if variant == 'librosa':
        W, H = librosa.decompose.decompose(magnitude, n_components=n_components, sort=True)
    else:
        pass

    return W, H

if __name__ == '__main__':
    imput = 'test_Pathway020.mp3'
    file_name, file_extension = imput.split('.')
    path = './to_separate/' + imput

    n_instruments = 3

    audio_sample, sampling_rate = librosa.load(path)

    n_fft = 2048 #number of fft per unit of time
    hop_length = 512
    spectrogram = librosa.stft(audio_sample, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(spectrogram)

    W, H = decompose(magnitude, n_instruments, variant='librosa')

    V = np.dot(W, H)
    V = np.multiply(V, phase)
    recovered_signal = librosa.istft(V, n_fft=n_fft, hop_length=hop_length)

    os.mkdir(f'./separated/{file_name}/')
    sf.write(f'./separated/{file_name}/recovered.wav', recovered_signal, sampling_rate)
