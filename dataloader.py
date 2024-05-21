import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit
import torch.nn as nn

# Assuming the fft and lp functions are defined elsewhere
from fft import lp, fourier_transform

def apply_pca(eeg_data, n_components):
    """
    Applies PCA to reduce the dimensionality of EEG data.
    
    :param eeg_data: (n_samples, n_features, sequence_length)
    :param n_components: int
    :return: Transformed EEG data with reduced dimensions
    """
    num_samples, num_channels, sequence_length = eeg_data.shape
    reshaped_data = eeg_data.transpose(0, 2, 1).reshape(-1, num_channels)  # Reshape for PCA
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(reshaped_data)
    reduced_data = reduced_data.reshape(num_samples, sequence_length, n_components).transpose(0, 2, 1)  # Reshape back
    return reduced_data

def make_windows_with_fft(pca_eeg, joint_data, eeg_window_size, joint_window_size, offset, lowpass_cutoff=49, debug=False):
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

def create_time_series_dataloaders(data, labels, batch_size, n_splits=5):
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


# Example usage with simulated data
num_samples = 1000
num_channels = 60
sequence_length = 100  # Number of time steps in each sequence
target_dimension = 4
n_components = 10  # Number of components to keep after PCA
batch_size = 32

# Simulate EEG data (num_samples, num_channels, sequence_length)
eeg_data = np.random.rand(num_samples, num_channels, sequence_length)

# Simulate target angular motion velocity (num_samples, target_dimension)
target_data = np.random.rand(num_samples, target_dimension)

# Apply PCA
eeg_data_pca = apply_pca(eeg_data, n_components)

# Create windows
eeg_window_size = 50
joint_window_size = 10
offset = 5
data, labels = make_windows_with_fft(eeg_data_pca, target_data, eeg_window_size, joint_window_size, offset, debug=True)

# Create DataLoader
train_loader, test_loader = create_time_series_dataloaders(data, labels, batch_size, n_splits=5)

# Create LSTM model
input_size = n_components
hidden_size = 64
num_layers = 2
output_size = target_dimension
model = LSTM_model_maker(input_size, hidden_size, num_layers, output_size)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model
num_epochs = 20
train_model(model, train_loader, num_epochs, criterion, optimizer)