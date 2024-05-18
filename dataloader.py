import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from fft import lp, fourier_transform

print("hello")


def make_windows_with_fft(pca_eeg, joint_data, eeg_window_size, joint_window_size, offset, lowpass_cutoff=49, debug=False):
    """
    :param pca_eeg: (n_samples, n_features)
    :param joint_data: (n_samples, n_features)
    :param window_size: int
    :param offset: int
    :param debug: bool
    :return: (n_windows, window_size, n_features)
    """
    for channel in pca_eeg:
        channel = lp(channel, lowpass_cutoff)

    n_samples = pca_eeg.shape[0]
    n_features = pca_eeg.shape[1]
    eeg_windows = [pca_eeg[i: i + eeg_window_size][:] for i in range(0, n_samples - eeg_window_size + 1, eeg_window_size)]
    joint_windows = [joint_data[i: i + joint_window_size][:] for i in range(eeg_window_size + offset, n_samples - joint_window_size + 1, eeg_window_size)]

    eeg_windows = eeg_windows[:len(joint_windows)]
    n_windows = len(joint_windows)

    if debug:
        print("n_samples: ", n_samples)
        print("n_features: ", n_features)
        print("n_windows: ", n_windows)
        print("len(eeg_windows): ", len(eeg_windows))
        print("len(joint_windows): ", len(joint_windows))

    assert(len(eeg_windows) == len(joint_windows))

    data = []
    labels = []

    if debug:
        print("eeg_windows[0].shape", eeg_windows[0].shape)
        print("joint_windows[0].shape", joint_windows[0].shape)

    for i in range(n_windows):

        pairs_per_channel = []
        for channel in range(n_features):
            window = eeg_windows[i]
            trans_window = window.T
            input = trans_window[channel][:]
            pairs = fourier_transform(trans_window[channel][:], 100)
            pairs_per_channel.append(pairs)
        this_window_data = np.array(pairs_per_channel)
        average_joint = np.mean(joint_windows[i], axis=0)

        data.append(this_window_data)
        labels.append(average_joint)

    return data, labels


def make_data_loader(data, labels, batch_size, shuffle=True):
    class CustomDataset(Dataset):
        def __init__(self, data, labels):
            self.data = data
            self.labels = labels
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            sample = self.data[idx]
            label = self.labels[idx]
            return sample, label
        
    dataset = CustomDataset(data, labels)

    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

