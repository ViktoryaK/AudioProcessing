import librosa
import numpy as np
import soundfile as sf
import os, shutil
import time
from separator import euclidian_distance
# from flask import Flask, render_template, url_for, redirect, request

def SNR(original: np.ndarray, recovered: np.ndarray):
    original_sliced = original[:recovered.shape[0]]
    numerator = np.sum(original_sliced ** 2)
    denominator = np.sum((original_sliced - recovered) ** 2)
    return numerator / denominator

if __name__ == "__main__":
    original, sr = librosa.load('./to_separate/test_Pathway.wav')
    recovered0, _ = librosa.load('./separated/test_Pathway/recovered(0).wav')
    recovered1, _ = librosa.load('./separated/test_Pathway/recovered(1).wav')

    print('SIGNAL TO NOISE RATIO')
    print('Signal recovered with product:', SNR(original, recovered0))
    print('Signal recovered with phase multiplication:', SNR(original, recovered1))