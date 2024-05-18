import scipy.signal as sig
import numpy as np


def butter_lowpass(cutoff, fs, order=5):
    return sig.butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = sig.lfilter(b, a, data)
    return y

def lp(data, fc): # filter out all freq above 50 with 
    return butter_lowpass_filter(data=data, cutoff=fc, fs=100)

def fourier_transform(data, fs):
    n = len(data)
    f = np.fft.fftfreq(n, 1/fs)
    y = np.fft.fft(data)
    
    f = np.fft.fftshift(f)
    y = np.fft.fftshift(y)

    y = np.abs(y)

    assert(len(f) == len(y))
    pairs = list(zip(f, y))
    # new_pairs = []
    # for pair in pairs:
    #     new_pair = (abs(pair[0]), pair[1])
    #     if new_pair not in new_pairs:
    #         new_pairs.append(new_pair)

    # pairs = new_pairs
    pairs.sort(key=lambda x: x[0])

    # print(pairs)
    output = np.zeros(shape=(len(pairs), 2))
    for i in range(len(pairs)):
        output[i][0] = pairs[i][0]
        output[i][1] = pairs[i][1]
    return output
    # return np.ndarray(pairs)