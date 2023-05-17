import argparse
import subprocess
from math import inf, log
import matplotlib.pyplot as plt
import numpy as np
from statistics import stdev
from separator import decompose
import librosa
import time



#signal-noise-ratio

def execute_decomposition_n_times(number_of_components):
    times = [[], []] #lib, our, long
    components = []
    name_input = 'PathwayMono.wav'
    path = './to_separate/' + name_input

    n_instruments = 3

    audio_sample, sampling_rate = librosa.load(path)

    n_fft = 2048  # number of fft per unit of time
    hop_length = 512
    spectrogram = librosa.stft(audio_sample, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(spectrogram)

    for n in range(number_of_components):
        start = time.time()
        decompose(magnitude, n_instruments, number_of_components, variant='librosa')
        end = time.time()
        times[0].append(end-start)
        start = time.time()
        decompose(magnitude, n_instruments, number_of_components, variant='euclidian')
        end = time.time()
        times[1].append(end - start)
        # start = time.time()
        # decompose(magnitude, n_instruments, number_of_components, variant='divergence')
        # end = time.time()
        # times[2].append(end - start)
        components.append(n)
        print(n)
    return times, components

def build_graph(time, components):
    br1 = np.arange(len(components))
    plt.plot(br1, time[0], color='darkviolet', label='librosa')
    plt.plot(br1, time[1], color='gold', label='euclidian')
    # plt.plot(br1, time[2], color='darkcyan', label='divergence')
    plt.xlabel("Number of components per instrument")
    plt.ylabel("Execution time")
    plt.xticks([r for r in range(len(components))], components)
    plt.title(f"Time dependency")
    plt.legend()
    plt.savefig(f"time_dependency.pdf")
    plt.cla()



def snr(x, r):
    numerator = 0
    for t in range(len(x)):
        numerator += x[t]**2
    denominator = 0
    for t in range(len(x)):
        denominator += (x[t]-r[t])**2

    snr = 10*log(numerator/denominator, 10)
    return snr


if __name__ == "__main__":
    # times, components = execute_decomposition_n_times(25)
    # build_graph(times, components)
    x_path = "./to_separate/test_Pathway.wav"
    r0_path = "./separated/test_Pathway/recovered(0).wav"
    r1_path = "./separated/test_Pathway/recovered(1).wav"
    x_audio_sample, x_sampling_rate = librosa.load(x_path)
    r0_a_s, r0_s_r = librosa.load(r0_path)
    r1_a_s, r1_s_r = librosa.load(r1_path)
    print(snr(x_audio_sample, r0_a_s))
    print(snr(x_audio_sample, r1_a_s))