import librosa
import numpy as np
import soundfile as sf
import os, shutil


def euclidian_distance(V: np.ndarray, W: np.ndarray, H: np.ndarray):
    dist = 0
    product = np.matmul(W, H)
    for i in range(len(V)):
        for j in range(len(V[0])):
            dist += (V[i][j] - product[i][j])**2

    return dist


def divergence(V: np.ndarray, W: np.ndarray, H: np.ndarray):
  diver = 0
  product = np.matmul(W, H)
  for i in range(len(V)):
      for j in range(len(V[0])):
          diver += (
              V[i][j] * np.log(V[i][j] / product[i][j]) -
              V[i][j] + product[i][j]
          )

  return diver


def decompose(V: np.ndarray, n_instruments: int, variant='euclidian', cpi=4, number_of_iterations=20, threshold=1e-4):
    n_components = n_instruments * cpi
    W = np.random.uniform(0, 1, (V.shape[0], n_components))
    H = np.random.uniform(0, 1, (n_components, V.shape[1]))

    if variant == 'librosa':
        W, H = librosa.decompose.decompose(V, n_components=n_components, sort=True)

    elif variant == 'euclidian':
        distance = euclidian_distance(V, W, H)
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
            new_distance = euclidian_distance(V, W, H)
            if (distance-new_distance) >= 0 and (distance-new_distance <= threshold):
                break

    elif variant == 'divergence':
        V = np.float64(V)
        diver = divergence(V, W, H)

        for iteration in range(number_of_iterations):
            W_new, H_new = W.copy(), H.copy()
            WH = W@H
            for i in range(H.shape[0]):
                for j in range(H.shape[1]):
                    numerator = 0
                    for row in range(V.shape[0]):
                        numerator += W[row][i]*V[row][j] / (WH)[row][j]
                    denominator = 0
                    for row in range(W.shape[0]):
                        denominator += W[row][i]
                    H_new[i][j] = H[i][j] * numerator / denominator
            for i in range(W.shape[0]):
                for j in range(W.shape[1]):
                    numerator = 0
                    for column in range(V.shape[1]):
                        numerator += H[j][column]*V[i][column] / (WH)[i][column]
                    denominator = 0
                    for column in range(H.shape[1]):
                        denominator += H[j][column]
                    W_new[i][j] = W[i][j] * numerator / denominator
            W, H = W_new, H_new
            new_diver = divergence(V, W, H)
            if (diver-new_diver) >= 0 and (diver-new_diver <= threshold):
                break

    return W, H


def get_component(W: np.ndarray, H: np.ndarray):
    for i in range(W.shape[1]):
        yield np.dot(W[:, i][np.newaxis].T, H[i][np.newaxis])


def write_components(W: np.ndarray, H: np.ndarray, phase: np.ndarray):
    iterations = W.shape[1]
    components = get_component(W, H)
    recovery_spec = np.full((W@H).shape, 0)

    for i in range(iterations):
        component = next(components)
        component = np.multiply(component, phase)
        signal = librosa.istft(component, n_fft=n_fft, hop_length=hop_length)

        sf.write(f'./separated/{file_name}/component{i+1}.wav', signal, sampling_rate)

        recovery_spec = np.add(recovery_spec, component)

    return librosa.istft(recovery_spec, n_fft=n_fft, hop_length=hop_length)


if __name__ == '__main__':
    name_input = 'test_Pathway.wav'
    file_name, file_extension = name_input.split('.')
    path = './to_separate/' + name_input

    n_instruments = 3
    cpi = 5 #components per instrument

    audio_sample, sampling_rate = librosa.load(path)

    n_fft = 2048 #number of fft per unit of time
    hop_length = 512
    spectrogram = librosa.stft(audio_sample, n_fft=n_fft, hop_length=hop_length)
    magnitude, phase = librosa.magphase(spectrogram)

    W, H = decompose(magnitude, n_instruments=n_instruments, variant='euclidian', cpi=cpi)

    try:
        os.mkdir(f'./separated/{file_name}')
    except FileExistsError:
        shutil.rmtree(f'./separated/{file_name}')
        os.mkdir(f'./separated/{file_name}')

    recovered_signal = write_components(W, H, phase)
    sf.write(f'./separated/{file_name}/recovered(1).wav', recovered_signal, sampling_rate)

    V = np.dot(W, H)
    V = np.multiply(V, phase)
    recovered_signal = librosa.istft(V, n_fft=n_fft, hop_length=hop_length)

    sf.write(f'./separated/{file_name}/recovered(0).wav', recovered_signal, sampling_rate)
