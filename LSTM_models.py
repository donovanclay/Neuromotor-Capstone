import numpy as np
import xarray as xr
import Functions_model2 as FM
from sklearn.model_selection import TimeSeriesSplit
import torch.nn as nn
import torch.optim as optim
import torch
import torch.nn.functional as F
import scipy.signal as signal
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import os
from sklearn.decomposition import FastICA
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Loading data for testing
Windows_path='C:\\Users\\heather\\Desktop\\Neuromotor\\Data4ML'
eeg_path=Windows_path+'\\3_sessions_eeg_filtered'
joint_path=Windows_path+'\\3_sessions_joint'
velocity_path=Windows_path+'\\3_session_velocity'
# Selects only the first trial for test run
eeg_data=xr.load_dataset(eeg_path)
eeg_=eeg_data.isel(Trial=0).copy()
eeg=eeg_.__xarray_dataarray_variable__
eeg=np.array(eeg.values)

joint_data=xr.load_dataset(joint_path)
joint_=joint_data.isel(Trial=0).copy()
joint=joint_.__xarray_dataarray_variable__
joint=np.array(joint.values)

velocity_data=xr.load_dataset(velocity_path)
velocity_=velocity_data.isel(Trial=0).copy()
velocity=velocity_.__xarray_dataarray_variable__
velocity=np.array(velocity.values)

def joint_2_velocity(joint_angles,fs): 
    velocity=np.diff(joint_angles, axis=0)*100
    return velocity

velocity_joint=joint_2_velocity(joint,100)

eeg,velocity_joint=FM.align_data_by_offset_joint(eeg,velocity_joint,4)
eeg,velocity=FM.align_data_by_offset(eeg,velocity,4)
ica=FastICA(n_components=25)
eeg=ica.fit_transform(eeg)
print('After ICA the shape of the EEG matrix is', eeg.shape)

velocity_joint=FM.lp(velocity_joint,0.5)
velocity=FM.lp(velocity,0.05)

velocity_joint,scalerj =FM.velocity_rescaler(velocity_joint)
velocity,scalerV =FM.velocity_rescaler(velocity)

for i in range(velocity.shape[1]):
    velocity[:, i] = FM.filter_and_interpolate(velocity[:, i])

stft_data,t= FM.apply_stft(eeg)
stft_data_reshaped = stft_data.transpose(2, 0, 1) # reshape the STFT data
X = stft_data_reshaped # input data
y = FM.velocity_windower(velocity,t)
print(f'X: {X.shape}, y: {y.shape},  time: {t.shape}')

X_sp=FM.split_sequences(X, 6) # this allows us to take into account  many time steps

y_sp=FM.generate_y(y,6)
print(f'X: {X_sp.shape}, y: {y_sp.shape}')

total_samples = len(X)
train_test_cutoff = round(0.90 * total_samples)

X_train = X_sp[:-train_test_cutoff]
X_test = X_sp[-train_test_cutoff:]

y_train = y_sp[:-train_test_cutoff]
y_test = y_sp[-train_test_cutoff:] 

np.save('X_train.npy', X_train)
np.save('X_test.npy', X_test)
np.save('y_train.npy', y_train)
np.save('y_test.npy', y_test)

print("Training Shape:", X_train.shape, y_train.shape)
print("Testing Shape:", X_test.shape, y_test.shape) 

