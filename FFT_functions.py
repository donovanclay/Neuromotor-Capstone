import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import scipy.signal as sig
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit


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
    pairs.sort(key=lambda x: x[0])
    output = np.zeros(shape=(len(pairs), 2))
    for i in range(len(pairs)):
        output[i][0] = pairs[i][0]
        output[i][1] = pairs[i][1]
    return output


def make_windows_with_fft(ca_eeg, joint_data, eeg_window_size, joint_window_size, offset, lowpass_cutoff=49, debug=False):
    """
    :param pca_eeg: (n_samples, n_features)
    :param joint_data: (n_samples, n_features)
    :param window_size: int
    :param offset: int
    :param debug: bool
    :return: (n_windows, window_size, n_features)
    """
    for channel in ca_eeg:
        channel = lp(channel, lowpass_cutoff)

    n_samples = ca_eeg.shape[0]
    n_features = ca_eeg.shape[1]
    eeg_windows = [ca_eeg[i: i + eeg_window_size][:] for i in range(0, n_samples - eeg_window_size + 1, eeg_window_size)]
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


def make_data_loader(data, labels, batch_size, shuffle=False):
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


def create_time_series_dataloaders(data, labels, batch_size, n_splits=5):
    """
    Creates DataLoader objects for time series data with time-based splitting.

    Args:
    - data (array-like): The time series data.
    - labels (array-like): The corresponding labels.
    - batch_size (int): The batch size for DataLoader objects.
    - n_splits (int): The number of splits for time series splitting (default: 5).

    Returns:
    - dataloaders (list): A list of DataLoader objects for each split.
    - train_indices_list (list): A list of train indices for each split.
    - test_indices_list (list): A list of test indices for each split.
    - train_data_list (list): A list of train data arrays for each split.
    - test_data_list (list): A list of test data arrays for each split.
    - train_labels_list (list): A list of train label arrays for each split.
    - test_labels_list (list): A list of test label arrays for each split.
    """
    class CustomTimeSeriesDataset(Dataset):
        def __init__(self, data, labels, train_indices, test_indices):
            self.train_data = [data[i] for i in train_indices]
            self.train_labels = [labels[i] for i in train_indices]
            self.test_data = [data[i] for i in test_indices]
            self.test_labels = [labels[i] for i in test_indices]

        def __len__(self):
            return len(self.train_data) + len(self.test_data)

        def __getitem__(self, idx):
            if idx < len(self.train_data):
                return self.train_data[idx], self.train_labels[idx]
            else:
                test_idx = idx - len(self.train_data)
                return self.test_data[test_idx], self.test_labels[test_idx]
        
    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_indices_list = []
    test_indices_list = []
    train_data_list = []
    test_data_list = []
    train_labels_list = []
    test_labels_list = []

    for train_index, test_index in tscv.split(data):
        train_indices_list.append(train_index)
        test_indices_list.append(test_index)
        train_data_list.append([data[i] for i in train_index])
        test_data_list.append([data[i] for i in test_index])
        train_labels_list.append([labels[i] for i in train_index])
        test_labels_list.append([labels[i] for i in test_index])

    dataloaders = []
    for train_indices, test_indices in zip(train_indices_list, test_indices_list):
        dataset = CustomTimeSeriesDataset(data, labels, train_indices, test_indices)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        dataloaders.append(dataloader)

    return dataloaders, train_indices_list, test_indices_list, train_data_list, test_data_list, train_labels_list, test_labels_list

def LSTM_model_maker(input_size, hidden_size, num_layers, output_size):
    class LSTMModel(nn.Module):
        def __init__(self, input_size, hidden_size, num_layers, output_size):
            super(LSTMModel, self).__init__()
            self.hidden_size = hidden_size
            self.num_layers = num_layers
            self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_size)

        def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
            out, _ = self.lstm(x, (h0, c0))
            out = self.fc(out[:, -1, :])
            return out

    return LSTMModel(input_size, hidden_size, num_layers, output_size)

def train_model(model, train_loader, num_epochs, criterion, optimizer):
    model.train()
    for epoch in range(num_epochs):
        for i, (x, y) in enumerate(train_loader):
            x = x.view(x.size(0), -1, x.size(-1)).float()  
            y = y.view(-1, 1).float()  
            outputs = model(x)
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')