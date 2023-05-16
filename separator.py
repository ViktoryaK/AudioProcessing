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




def decompose(V: np.ndarray, n_instruments: int, variant='', number_of_iterations=20):
    n_components = n_instruments * 4

    if variant == 'librosa':
        W, H = librosa.decompose.decompose(V, n_components=n_components, sort=True)
    else:
        W = np.random.uniform(0, 1, (V.shape[0], n_components))
        H = np.random.uniform(0, 1, (n_components, V.shape[1]))
        for iteration in range(number_of_iterations):
            WTV = W.T@V
            WTWH = W.T@W@H
            for i in range(H.shape[0]):
                for j in range(H.shape[1]):
                    H[i][j] = H[i][j] * WTV[i][j] / WTWH[i][j]
            VHT = V@H.T
            WHHT = W@H@H.T
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    W[i][j] = W[i][j] * VHT[i][j] / WHHT[i][j]
    return W, H

if __name__ == '__main__':
    name_input = 'test_Pathway.wav'
    file_name, file_extension = name_input.split('.')
    path = './to_separate/' + name_input

    n_instruments = 3

    audio_sample, sampling_rate = librosa.load(path)

    n_fft = 2048 #number of fft per unit of time
    hop_length = 512
    spectrogram = librosa.stft(audio_sample, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(spectrogram)

    W, H = decompose(magnitude, n_instruments, variant='our')

    V = np.dot(W, H)
    V = np.multiply(V, phase)
    recovered_signal = librosa.istft(V, n_fft=n_fft, hop_length=hop_length)

    os.mkdir(f'./separated/{file_name}')
    sf.write(f'./separated/{file_name}/recovered.wav', recovered_signal, sampling_rate)
