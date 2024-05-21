import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import torch

def prepare_eeg_data_for_lstm(eeg_data, target_data, n_components, sequence_length, batch_size):
    """
    Prepares EEG data for LSTM by reducing dimensionality with PCA and creating DataLoaders.
    
    Parameters:
    eeg_data (numpy array): The EEG data of shape (num_samples, num_channels, sequence_length).
    target_data (numpy array): The target data of shape (num_samples, target_dimension).
    n_components (int): The number of principal components to keep after PCA.
    sequence_length (int): The number of time steps in each sequence.
    batch_size (int): The batch size for the DataLoader.
    
    Returns:
    train_loader (DataLoader): DataLoader for the training set.
    test_loader (DataLoader): DataLoader for the test set.
    """
    num_samples, num_channels, _ = eeg_data.shape
    
    # Reshape EEG data for PCA (num_samples * sequence_length, num_channels)
    eeg_data_reshaped = eeg_data.transpose(0, 2, 1).reshape(-1, num_channels)
    
    # Apply PCA
    pca = PCA(n_components=n_components)
    eeg_data_reduced = pca.fit_transform(eeg_data_reshaped)
    
    # Reshape back to (num_samples, sequence_length, n_components)
    eeg_data_reduced = eeg_data_reduced.reshape(num_samples, sequence_length, n_components)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(eeg_data_reduced, target_data, test_size=0.2, random_state=42)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader



# Example usage
num_samples = 1000
num_channels = 60
sequence_length = 100  # Number of time steps in each sequence
target_dimension = 4
n_components = 10  # Number of components to keep after PCA
batch_size = 32

# Simulate EEG data (num_samples, num_channels, sequence_length)
eeg_data = np.random.rand(num_samples, num_channels, sequence_length)

# Simulate target angular motion velocity (num_samples, target_dimension)
target_data = np.random.rand(num​⬤