import matplotlib.pyplot as plt
import numpy as np
from separator import decompose
import librosa
import time

def execute_decomposition_n_times(number_of_components):
    times = [[], [], []]
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
        start = time.time()
        decompose(magnitude, n_instruments, number_of_components, variant='divergence')
        end = time.time()
        times[2].append(end - start)
        components.append(n)
        print(n)
    return times, components

def build_graph(time, components):
    br1 = np.arange(len(components))
    plt.plot(br1, time[0], color='darkviolet', label='librosa')
    plt.plot(br1, time[1], color='gold', label='euclidian')
    plt.plot(br1, time[2], color='darkcyan', label='divergence')
    plt.xlabel("Number of components per instrument")
    plt.ylabel("Execution time")
    plt.xticks([r for r in range(len(components))], components)
    plt.title(f"Time dependency")
    plt.legend()
    plt.savefig(f"time_dependency.pdf")
    plt.cla()

if __name__ == "__main__":
    times, components = execute_decomposition_n_times(10)
    build_graph(times, components)
