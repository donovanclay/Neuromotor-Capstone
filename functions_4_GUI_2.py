import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xarray import DataArray
from bs4 import BeautifulSoup 
import os
import warnings
from statsmodels.robust.scale import huber
from mne._fiff.pick import _MEG_CH_TYPES_SPLIT
from mne.utils import (fill_doc, _check_sphere,_validate_type,
                        _check_option,logger) #take out last
from mne.defaults import (
    _BORDER_DEFAULT,
    _EXTRAPOLATE_DEFAULT,
    _INTERPOLATION_DEFAULT)

from PyQt5.QtCore import QtInfoMsg, QtWarningMsg, QtCriticalMsg
# comment out windows path as needed
Windows_path='/mnt/c/Users/Heather/Desktop/Neuromotor/RepositoryData'
path_all = '../RepositoryData'
path_all= Windows_path # Comment out as needed



def electrode_labels(Patient_numb='/SL01'):
    sgl_ele_labels= path_all +Patient_numb+'-T01'+'/impedances-before.txt'
    # Remove the first 2 lines of the file using skiprows
    ele_imp = pd.read_csv(sgl_ele_labels, sep='\t', header=None , skiprows=20, names=list(range(5)))
    ele_labels=ele_imp[1]
    ele_labels=ele_labels.drop(0)
    return ele_labels

    

def drop_excluded_EoG(Patient_numb='/SL01'): 
    ele_labels=electrode_labels(Patient_numb='/SL01')
    EOG_labels=[' TP9', 'TP10', ' FT9', 'FT10']
    EOG_labels=ele_labels[ele_labels.iloc[:].isin(EOG_labels)]
    EOG_indx=EOG_labels.index
    return EOG_indx

def normalize_eeg(eeg_walk):
    for i in range(eeg_walk.shape[1]):
        Hmean,Hstd= huber(eeg_walk.iloc[:, i])
        HCentered = eeg_walk.iloc[:, i] - Hmean
        Hnorm = HCentered / Hstd
        eeg_walk.iloc[:, i] = Hnorm
    return eeg_walk

# functions to be called from GUI
def Eeg_data_function(Patient_numb='/SL01', trial_numbers=[1,2,3], time_interval=[2,17], frequency=100, drop_excluded=True):
    ''' Loads eeg data from specified patient and returns an xarray
    Inputs: 
    Patient_numb: string example '/SL01'
    trial_numbers: array example [1] which would be trial 1 or  [2,3] (trials 2 & 3)
    time_interval: array containing time interval in minutes [beginning, end]
    frequency: integer in s^-1
    Outputs:
    eeg_data: xarray
    '''
    # Load the data
    if drop_excluded==True:
        length=60 # might be a better_way to get this
    else:
        length=64
    fs = frequency
    Trial_path=['-T01/eeg.txt','-T02/eeg.txt','-T03/eeg.txt']  # Corrected path for different trials
    Eog_index= drop_excluded_EoG(Patient_numb)
    eeg_data = np.empty((len(trial_numbers), 90000, length), dtype=float)  # Initialize eeg_data as multi-dimensional array
    for i, trial in enumerate(trial_numbers):
        eeg = pd.read_csv(path_all + Patient_numb + Trial_path[trial-1], sep='\t', on_bad_lines='warn', skiprows=1, header=None)
        eeg.dropna(axis=1, how='all', inplace=True)
        # first column is the time in seconds
        time = eeg.iloc[:, 0]
        eeg = eeg.iloc[:, 1:]
        # drops unwanted sections
        eeg_walk = eeg.drop(range(time_interval[0]*60*fs))
        eeg_walk = eeg_walk.drop(range(time_interval[1]*60*fs-1, eeg_walk.index[-1]))
        if drop_excluded==True:
            for EoG in Eog_index:
                eeg_walk.drop(eeg_walk.columns[EoG], axis=1, inplace=True)
        normalize_eeg(eeg_walk)
        eeg_walk = eeg_walk.to_numpy()
        eeg_data[i] = eeg_walk  # Assigning to the correct slice of eeg_data
    timew= time.drop(range(time_interval[0]*60*fs))
    timew = timew.drop(range(time_interval[1]*60*fs-1, timew.index[-1]))
    timew = timew.to_numpy()

    eeg_data = DataArray(
            eeg_data,
            dims=["Trial", "Recording", "Channel"],
            coords={
                "Trial": trial_numbers,
                "Recording": np.arange(eeg_data.shape[1]),
                "Channel": np.arange(eeg_data.shape[2]),
                "Time": ( "Recording", timew),  # Include time as a coordinate
            },
        )
    return eeg_data

# patient_1 = Eeg_data_function()
# new=patient_1.drop_isel(Channel=-1)
# print(new.shape,patient_1.shape)


def joint_data_function(Patient_numb='/SL01', trial_numbers=[1], time_interval=[2,17], frequency=100):
    ''' Loads eeg data from specified patient and returns an xarray
    Inputs: 
    Patient_numb: string example '/SL01'
    trial_numbers: array example [1] which would be trial 1 or  [2,3] (trials 2 & 3)
    time_interval: array containing time interval in minutes [beginning, end]
    frequency: integer in s^-1
    Outputs:
    joint_data: xarray
    '''
    # Load the data
    fs = frequency
    Trial_path=['-T01/joints.txt','-T02/joints.txt','-T03/joints.txt']  # Corrected path for different trials
    j_data = np.empty((len(trial_numbers), 90000, 6), dtype=float)  # Initialize eeg_data as multi-dimensional array
    for i, trial in enumerate(trial_numbers):
        # Remove the first 2 lines of the file using skiprows
        j_angl = pd.read_csv(path_all + Patient_numb+Trial_path[trial-1], sep='\t', header=None ,names=list(range(14)), skiprows=2)
        # processing data
        j_angl.dropna(axis=1, how='all', inplace=True)
        # first column is the time in seconds
        j_time = j_angl.iloc[:, 0]
        j_angl = j_angl.drop(0, axis=1)
        j_angl = j_angl.drop(range(7,13), axis=1) # we only want actual joint angles for testing 7-13 are predictions
        # drops unwanted sections
        j_walk = j_angl.drop(range(time_interval[0]*60*fs))
        j_walk = j_walk.drop(range(time_interval[1]*60*fs-1, j_walk.index[-1]))
        j_walk = j_walk.to_numpy()
        j_data[i] = j_walk  
    timew= j_time.drop(range(time_interval[0]*60*fs))
    timew = timew.drop(range(time_interval[1]*60*fs-1, timew.index[-1]))
    timew = timew.to_numpy()
    joint_data = DataArray(
            j_data,
            dims=["Trial", "Recording", "Channel"],
            coords={
                "Trial": trial_numbers,
                "Recording": np.arange(j_data.shape[1]),
                "Channel": np.arange(j_data.shape[2]),
                "Time": ( "Recording", timew),  # Include time as a coordinate
            },
        )
    return joint_data

# joint testing
# joint1=joint_data_function()
#print(joint1[0].values.shape)

def spatial_data_function(Patient_numb='/SL01', trial_numbers=[1]):
    ''' Loads eeg data from specified patient and returns an xarray
    Inputs: 
    Patient_numb: string example '/SL01'
    frequency: integer in s^-1
    Outputs:
    eeg_data: xarray
    '''
    # Load the data
    Trial_path=['-T01/digitizer.bvct','-T02/digitizer.bvct','-T03/digitizer.bvct']  # Corrected path for different trials
    spatial_data = []  # Initialize spatial_data as a list
    for ind, trial in enumerate(trial_numbers):
        cols = ['Name', 'X', 'Y', 'Z', 'Theta', 'Phi', 'Radius', 'Channel'] 
        rows = [] 
        # Opening file
        file = open(path_all + Patient_numb + Trial_path[trial-1], 'r') 
        contents = file.read()
        soup = BeautifulSoup(contents, 'xml')
        Name = soup.find_all('Name')
        X = soup.find_all('X')
        Y = soup.find_all('Y')
        Z = soup.find_all('Z')
        Theta = soup.find_all('Theta')
        Phi = soup.find_all('Phi')
        Radius = soup.find_all('Radius')
        Channel = soup.find_all('Channel')
        # Set data to none
        data = [] 
        for i in range(len(Name)): 
            rows = [Name[i].get_text(), float(X[i].get_text()), 
                    float(Y[i].get_text()), float(Z[i].get_text()), 
                    float(Theta[i].get_text()), float(Phi[i].get_text()),
                    float(Radius[i].get_text()), float(Channel[i].get_text())]  
            data.append(rows)
        if ind == 0:
            spatial_data = np.array(data)
            sum_data = np.empty((len(trial_numbers), spatial_data.shape[0], spatial_data.shape[1]), dtype=object)
            sum_data[ind] = np.array(data)
        else:
            New_data = np.array(data)
            sum_data[ind] = np.array(data)
    # Stack along the first dimension outside the loop
    spatial_data = np.stack(sum_data, axis=0)

    spatial_data = DataArray(
            spatial_data,
            dims=["Trial", "Electrode", "Values"],
            coords={
                "Trial": trial_numbers,
                "Electrode": np.arange(spatial_data.shape[1]),
                "Values": np.arange(spatial_data.shape[2]),
            },
        )
    return spatial_data

# Spatial testing
patient_spatial=spatial_data_function()
electrodes=patient_spatial.sel(Values=0)
electrodes = electrodes.values  # Convert DataArray to numpy array
electrodes=electrodes[0]
electrodes_2 = electrode_labels().values  # Convert DataArray to numpy array
# Strip spaces from electrode labels
electrodes_2 = np.array([ele.strip() for ele in electrodes_2])
print(electrodes.shape, electrodes_2.shape)
drop_electrodes = np.array([])
for electrode in electrodes:
    if electrode not in electrodes_2:
        drop_electrodes = np.append(drop_electrodes, electrode)

print(drop_electrodes)
# one_sel=patient_spatial.sel(Values=1).astype(float).to_numpy()
# print(one_sel[0][:])

def get_EoG_index():
        # Loading electrode labels
        ele_imp = pd.read_csv(path_all+'/SL01-T01/impedances-before.txt', sep='\t', header=None , skiprows=2, names=list(range(5)))
        ele_labels=ele_imp[1]
        ele_labels=ele_labels.drop(0)
        EOG_labels=[' TP9', 'TP10', ' FT9', 'FT10']
        EOG_labels=ele_labels[ele_labels.iloc[:].isin(EOG_labels)]
        return EOG_labels.index

