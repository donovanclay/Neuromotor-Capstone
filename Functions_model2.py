import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import scipy.signal as sig
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from torch.nn.utils.rnn import pad_sequence


def velocity_rescaler(data):
    scalers = []
    for i in range(data.shape[1]):
        min_max_scaler = MinMaxScaler()
        channel = data[:, i]
        data_2d = channel.reshape(-1, 1)
        normalizedData = min_max_scaler.fit_transform(data_2d)
        normalized_data = normalizedData.flatten()
        data[:, i] = normalized_data
        scalers.append(min_max_scaler)
    return data, scalers

def undo_velocity_rescaler(data, scalers):
    for i in range(data.shape[1]):
        min_max_scaler = scalers[i]
        normalized_data = data[:, i]
        data_2d = normalized_data.reshape(-1, 1)
        originalData = min_max_scaler.inverse_transform(data_2d)
        original_data = originalData.flatten()
        data[:, i] = original_data
    return data

def align_data_by_offset_joint(eeg_data,velocity_data,offset):
    velocity=velocity_data[offset-1:]
    eeg=eeg_data[:-offset]
    return eeg,velocity


def align_data_by_offset(eeg_data,velocity_data,offset):
    velocity=velocity_data[offset:]
    eeg=eeg_data[:-offset]
    return eeg,velocity

def butter_lowpass(cutoff, fs, order=5):
    return sig.butter(order, cutoff, fs=fs, btype='low', analog=False)

def butter_lowpass_filter(data, cutoff, fs, order=9):
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


def make_windows_with_fft(ca_eeg, velocity_data, window_size, lowpass_cutoff=49, debug=False):
    """
    :param pca_eeg: (n_samples, n_features)
    :param joint_data: (n_samples, n_features)
    :param window_size: int
    :return: (n_windows, window_size, n_features)
    """
    for channel in ca_eeg:
        channel = lp(channel, lowpass_cutoff)

    n_samples = ca_eeg.shape[0]
    n_features = ca_eeg.shape[1]
    eeg_windows = [ca_eeg[i: i + window_size][:] for i in range(0, n_samples - window_size + 1, window_size)]
    joint_windows = [velocity_data[i: i + window_size][:] for i in range(0, n_samples - window_size + 1, window_size)]

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

def create_time_series(data, labels, train_indices, val_indices, test_indices, batch_size):
    class CustomTimeSeriesDataset(Dataset):
        def __init__(self, data, labels, indices):
            self.data = [data[i] for i in indices]
            self.labels = [labels[i] for i in indices]

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    def collate_fn(batch):
        data, labels = zip(*batch)
        data_padded = pad_sequence([torch.tensor(d) for d in data], batch_first=True, padding_value=0)
        labels = torch.tensor(labels)
        return data_padded, labels

    # Create the datasets
    train_dataset = CustomTimeSeriesDataset(data, labels, [i for indices in train_indices for i in indices])
    val_dataset = CustomTimeSeriesDataset(data, labels, [i for indices in val_indices for i in indices])
    test_dataset = CustomTimeSeriesDataset(data, labels, [i for indices in test_indices for i in indices])

    # Create DataLoader objects
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    return train_loader, val_loader, test_loader

def LSTM_model_maker(input_size, hidden_size, num_layers, output_size, drop_out=0.5):
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
            loss = criterion(outputs, y) # use this to plot loss over epochs
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')