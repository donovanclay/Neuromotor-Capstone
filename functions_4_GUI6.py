import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xarray import DataArray
from bs4 import BeautifulSoup 
import os
import warnings
from statsmodels.robust.scale import huber
import getpass
import subprocess
from scipy.stats import linregress
from mne.preprocessing import EOGRegression
import mne
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import xarray as xr
from matplotlib.offsetbox import AnchoredText

# commands to manually add path if needed (all wsl2)
Windows_path='/mnt/c/Users/heather/Desktop/Neuromotor/RepositoryData'
path_all = '../RepositoryData'
path_all= Windows_path # Comment out as needed

def get_windows_username():
    try:
        result = subprocess.run(['wslvar', 'USERNAME'], capture_output=True, text=True)
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            print(f"Error: {result.stderr.strip()}")
            return None
    except Exception as e:
        print(f"Error retrieving Windows username: {e}")
        return None
    
def get_desktop_path():
    if os.name == 'posix':  # Checking if the OS is POSIX (Linux or WSL)
        wsl_mount = '/mnt/'
        for root, dirs, files in os.walk(wsl_mount):
            if 'Users' in dirs:
                user_name=get_windows_username()
                if user_name:
                    user_path = os.path.join(root, 'Users', user_name, 'Desktop')
                    try:
                        if os.path.exists(user_path):
                            return user_path
                    except OSError as e:
                        print(f"Error accessing {user_path}: {e}")
                        continue
    else:
        return os.path.join(os.path.expanduser('~'), 'Desktop')

def set_file_path(over_all_path):
    global path_all  # Declare path_all as global before using it
    path_all = over_all_path 

def unpack_spatial(spatial):
            ch_names=spatial.sel(Values=0)[0].values
            x=spatial.sel(Values=1)[0].astype(float).to_numpy()
            y=spatial.sel(Values=2)[0].astype(float).to_numpy()
            z=spatial.sel(Values=3)[0].astype(float).to_numpy()
            x=x/(np.max(x))
            y=y/(np.max(y))
            z=z/(np.max(z))
            xyz=np.stack((x, y, z), axis=1)
            ch_pos= {key: coord for key, coord in zip(ch_names, xyz)}
            return ch_names.tolist(), xyz, ch_pos


def get_included_chan_names(drop_excluded=True):
    ch_names=electrode_labels()
    if drop_excluded==True:
        indices = get_EoG_index(ch_names)
        ch_names = np.delete(ch_names, indices)
    ref_labels = ['Ref', 'Gnd']
    ref_mask = np.isin(ch_names, ref_labels)
    ref_indx = np.where(ref_mask)[0]
    ch_names = np.delete(ch_names, ref_indx)
    return ch_names.tolist()

def electrode_labels(Patient_numb='/SL01'):
    sgl_ele_labels= path_all +Patient_numb+'-T01'+'/impedances-before.txt'
    ele_imp = pd.read_csv(sgl_ele_labels, sep='\t', header=None , skiprows=20, names=list(range(5)))
    ele_labels=ele_imp[1]
    ele_labels=ele_labels.drop(0)
    ch_name = [name.strip() for name in ele_labels]
    return np.array(ch_name)

def get_EoG_index(Patient_numb='/SL01'): 
    ele_labels=electrode_labels(Patient_numb='/SL01')
    EOG_labels = ['TP9', 'TP10', 'FT9', 'FT10']
    mask = np.isin(ele_labels, EOG_labels)
    EOG_indx = np.where(mask)[0]
    return EOG_indx

def apply_function_by_channel2(eeg_data, function, channels, **params):
        trial_numbers = get_trial_numbers(eeg_data)
        if len(trial_numbers) == 1:  # Handle single trial case
            modified_eeg_data = function(eeg_data, **params)
            if isinstance(modified_eeg_data, xr.DataArray):
                if 'Trial' not in modified_eeg_data.dims:
                    modified_eeg_data = modified_eeg_data.expand_dims(dim='Trial', axis=0)
            else:
                raise TypeError("The function did not return an xarray.DataArray") 
        return modified_eeg_data

def apply_function_by_channel(eeg_data, function, channels, **params):
    modified_eeg_data = eeg_data.copy()  # Make a copy to avoid modifying the original data
    for trial in modified_eeg_data['Trial']:
        for channel in channels:
            modified_channel_data = function(modified_eeg_data.sel(Trial=trial, Channel=channel), **params)
            modified_eeg_data.loc[{'Trial': trial, 'Channel': channel}] = modified_channel_data.values
    return modified_eeg_data

def apply_function_by_trial(eeg_data, function=None, **params):
    trial_numbers = get_trial_numbers(eeg_data)
    if len(trial_numbers) == 1:  # Handle single trial case
        modified_eeg_data = function(eeg_data, **params)
        if isinstance(modified_eeg_data, xr.DataArray):
            if 'Trial' not in modified_eeg_data.dims:
                modified_eeg_data = modified_eeg_data.expand_dims(dim='Trial', axis=0)
        else:
            raise TypeError("The function did not return an xarray.DataArray")
    else:  # Handle multiple trials
        modified_eeg_data = []
        for trial in trial_numbers:
            modified_trial_data = function(eeg_data.sel(Trial=trial), **params)
            modified_eeg_data.append(modified_trial_data)
        modified_eeg_data = xr.concat(modified_eeg_data, dim="Trial")
    return modified_eeg_data


def normalize_eeg(eeg_walk):
    Hmean,Hstd= huber(eeg_walk)
    HCentered = eeg_walk - Hmean
    Hnorm = HCentered / Hstd
    eeg_walk = Hnorm
    return eeg_walk

def drop_eog_channels(eeg_data):
    drop_index = get_EoG_index()
    modified_eeg_data = eeg_data.drop_sel(Channel=drop_index)
    # Update the 'Channel' coordinate
    modified_eeg_data['Channel'] = np.arange(len(modified_eeg_data['Channel']))
    return modified_eeg_data

def get_trial_numbers(eeg_data):
    """
    Extracts and returns the unique trial numbers from the eeg_data xarray DataArray.
    
    Parameters:
    eeg_data (xarray.DataArray): The EEG data array from which to extract trial numbers.
    
    Returns:
    list: A list of unique trial numbers.
    """
    trial_numbers = eeg_data.coords['Trial'].values
    if np.isscalar(trial_numbers):  # If trial_numbers is a scalar, make it a list
        trial_numbers = [trial_numbers]
    elif isinstance(trial_numbers, np.ndarray) and trial_numbers.ndim == 0:  # If it's a 0-d array, make it a list
        trial_numbers = [trial_numbers.item()]
    else:  # Ensure it's a list if it's already an array
        trial_numbers = trial_numbers.tolist()
    return trial_numbers

# functions to be called from GUI
def spatial_data_function(Patient_numb='/SL01', trial_numbers=[1], drop_ref=True):
    ''' Loads eeg data from specified patient and returns an xarray
    Inputs: 
    Patient_numb: string example '/SL01'
    frequency: integer in s^-1
    drop_ref: Drops grounding electrodes and refence electrodes if true
    Outputs:
    eeg_data: xarray
    '''
    # Load the data
    dropped_electrodes=['FCz', 'AFz', 'Nasion']
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
            if drop_ref==True:
                if Name[i].get_text() in dropped_electrodes:
                    pass
                else:
                    rows = [Name[i].get_text(), float(X[i].get_text()), 
                            float(Y[i].get_text()), float(Z[i].get_text()), 
                            float(Theta[i].get_text()), float(Phi[i].get_text()),
                            float(Radius[i].get_text()), float(Channel[i].get_text())]
                    data.append(rows)  
            else:
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


def remap_spatial_2_channels(Include_none=False):    
    names = get_included_chan_names(drop_excluded=False)
    spatial_data = spatial_data_function()
    ch_names, xyz, _ = unpack_spatial(spatial_data)
    EoG_index = get_EoG_index()
    index_AF3 = names.index('AF3') # Since coordinates are not given for EoG they are approximated using AF3 which is close to eyes
    AF3=xyz[index_AF3]
    index_AF4 = names.index('AF4')
    AF4=xyz[index_AF4]
    names_by_chan = []
    xyz_by_chan = []
    sub = 0  # accounts for non-existence of EoG channels
    for i, name in enumerate(names):
        if i in EoG_index:
            sub += 1
            if Include_none == False:
                pass
            else:
                names_by_chan.append(name) 
                if name=='FT9':
                    xyz2=[AF3[0]+0.5,AF3[1],AF3[2]-0.1]
                elif name=='FT10':
                    xyz2=[AF4[0]-0.5,AF4[1],AF4[2]-0.2]
                elif name=='TP9':
                    xyz2=[AF3[0]-0.5,AF3[1]-0.5, AF3[2]-0.1]
                elif name=='TP10':
                    xyz2=[AF4[0]+0.5,AF4[1]-0.5,AF4[2]-0.2]
                xyz_by_chan.append(xyz2) #approximate xyz eye location
        else:
            if name in ch_names:
                names_by_chan.append(name)
                xyz_by_chan.append(xyz[i-sub])
            if name == 'T7':  # about the same location as A1
                names_by_chan.append('T7')
                xyz_by_chan.append(xyz[i-sub])
            if name == 'T8':  # about the same location as A2
                names_by_chan.append('T8')
                xyz_by_chan.append(xyz[i-sub])
    return np.array(names_by_chan), np.array(xyz_by_chan)


def Eeg_data_function(Patient_numb='/SL01', trial_numbers=[1,2,3], time_interval=[2,17], frequency=100, drop_excluded=False):
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
    Eog_index= get_EoG_index(Patient_numb)
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

def mne_data_type_converter(eeg_data,ch_names,ch_types):
        info = mne.create_info(ch_names=ch_names,sfreq=100, ch_types=ch_types)
        if isinstance(eeg_data, np.ndarray):
            raw=eeg_data
        else:
            raw= np.array(eeg_data.values)
        raw=np.transpose(raw)  
        raw = mne.io.RawArray(raw, info)
        raw.set_eeg_reference('average', projection=True)
        return raw


def plot_components(eeg_data, sources, artifact_index):
    n_components = sources.shape[0]
    fig = Figure(figsize=(10, 2 * n_components))
    axs = fig.subplots(n_components, 1, sharex=True)
    
    for i in range(n_components):
        axs[i].plot(range(len(eeg_data)), sources[i], label=f'Component {i+1}')
        axs[i].legend(loc='upper right')
        if i in artifact_index:
            anchored_text = AnchoredText("Component Dropped", loc=2)
            axs[i].add_artist(anchored_text)
        if i < n_components - 1:
            axs[i].set_xticks([])  

    axs[-1].set_xlabel('Time series')
    fig.text(0.04, 0.5, 'Amplitude of component', va='center', rotation='vertical')

    plt.switch_backend('Qt5Agg')  # Ensure using the correct backend

    # Save the figure
    fig.savefig('ICA_analysis.png')

def get_montage(ch_pos):
        montage = mne.channels.make_dig_montage(
        ch_pos=ch_pos 
        )
        return montage

def EoG_filter(eeg_data, ch_names,ch_pos):
    EoG_indx=get_EoG_index()
    chan_type=['eeg']*64
    for EoG in EoG_indx:
        chan_type[EoG]='eog'
    raw=mne_data_type_converter(eeg_data,ch_names,chan_type)
    montage=get_montage(ch_pos)
    raw.set_montage(montage)
    weights = EOGRegression().fit(raw)
    raw_clean = weights.apply(raw, copy=True)
    data = np.transpose(raw_clean.get_data())
    return raw_clean, data, weights

def remove_artifacts(eeg_data, threshold=6, window_size=16):
    original_shape = eeg_data.shape
    eeg_data_np = eeg_data.values
    flattened_data = eeg_data_np.reshape(-1, eeg_data_np.shape[-1])
    for i in range(flattened_data.shape[0]):
        signal = flattened_data[i]
        artifact_indices = np.abs(signal) > threshold
        for idx in np.argwhere(artifact_indices):
            idx = idx[0]  # Extracting a single integer from idx
            start_idx = max(0, idx - window_size // 2)
            end_idx = min(len(signal), idx + window_size // 2)
            window_length = end_idx - start_idx
            if window_length > 1:  # Ensure enough data points for linear regression
                X = np.arange(start_idx, end_idx).reshape(-1, 1)
                y = signal[start_idx:end_idx]
                slope, intercept, _, _, _ = linregress(X.flatten(), y)
                predicted_artifact = slope * idx + intercept
                signal[idx] -= predicted_artifact
    eeg_data_np = flattened_data.reshape(original_shape)
    eeg_data = xr.DataArray(eeg_data_np)
    return eeg_data


# eeg_data = Eeg_data_function()
# eeg_data = apply_function_by_trial(eeg_data, function=drop_eog_channels)
# channels=eeg_data.Channel
# eeg_data = apply_function_by_channel(eeg_data, function=normalize_eeg,channels=channels)
# eeg_data = apply_function_by_channel(eeg_data,function=remove_artifacts,channels=channels)


def velocity_function(data, time, interval, length):
    joints = int(data.shape[1])
    side1 = (interval - 1) // 2
    side2 = side1 + 1
    size = len(range(side1, length - side2, side1))
    velocity = np.empty((length, joints), dtype=float)
    warnings.simplefilter('ignore', np.RankWarning)
    for j in range(joints):
        one_joint = data[:, j]
        for i in range(side1, length - side2, side1):
            joint_subset, time_subset = one_joint[i-side1:i+side2], time[i-side1:i+side2]
            try:
                m, _,_,_,_ = linregress(joint_subset, time_subset)
                velocity[i][j] = m
                last_valid_index = i
            except ValueError as e:
                if str(e) == "Cannot calculate a linear regression if all x values are identical":
                    last_valid_values = velocity[i, j]
                    distance_from_last_valid = i - last_valid_index
                    weight = 1 / (distance_from_last_valid + 1)  # Assign higher weight to closer points
                    velocity[i][j] = one_joint[last_valid_index] * weight
                else:
                    velocity[i][j] = np.nan  # If no valid values are available, assign NaN
                    print('Nan values in velocity data')
        for i in range(side1, length - side2, side1):
            if i == 0 or i == length - 1:  # No interpolation at endpoints
                velocity[i][j] = one_joint[i]
            else:
                # Linear interpolation with 2 points on each side
                t_left = time[i - side1]
                t_right = time[i + side1]
                t_current = time[i]
                v_left = one_joint[i - side1]
                v_right = one_joint[i + side2]
                if t_right == t_left:
                    velocity[i][j] = (v_left + v_right) / 2  # Average of neighboring velocities
                else:
                    # Interpolate velocity at time i
                    velocity[i][j] = ((t_right - t_current) * v_left + (t_current - t_left) * v_right) / (t_right - t_left)
    return velocity

def joint_data_function(Patient_numb='/SL01', trial_numbers=[1], time_interval=[2,17], frequency=100, inter=10):
    ''' Loads eeg data from specified patient and returns an xarray
    Inputs: 
    Patient_numb: string example '/SL01'
    trial_numbers: array example [1] which would be trial 1 or  [2,3] (trials 2 & 3)
    time_interval: array containing time interval in minutes [beginning, end]
    frequency: integer in s^-1
    int: must be odd, interval velocity is fitted to
    Outputs:
    joint_data: xarray
    '''
    # Load the data
    length=90000
    interval=inter
    fs = frequency
    Trial_path=['-T01/joints.txt','-T02/joints.txt','-T03/joints.txt']  # Corrected path for different trials
    j_data = np.empty((len(trial_numbers), 90000, 6), dtype=float)  # Initialize eeg_data as multi-dimensional array
    velc_data=np.empty((len(trial_numbers), 90000, 6), dtype=float) 
    j_angl = pd.read_csv(path_all + Patient_numb+Trial_path[0], sep='\t', header=None ,names=list(range(14)), skiprows=2)
    j_angl.dropna(axis=1, how='all', inplace=True)
    j_time = j_angl.iloc[:, 0]
    timew= j_time.drop(range(time_interval[0]*60*fs))
    timew = timew.drop(range(time_interval[1]*60*fs-1, timew.index[-1]))
    timew = timew.to_numpy()
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
        velc_data[i] = velocity_function(j_walk,timew, interval, length)
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
    velocity_data = DataArray(
            velc_data,
            dims=["Trial", "Recording", "Channel"],
            coords={
                "Trial": trial_numbers,
                "Recording": np.arange(velc_data.shape[1]),
                "Channel": np.arange(velc_data.shape[2])})
    return joint_data, velocity_data

def spatial_data_function(Patient_numb='/SL01', trial_numbers=[1], drop_ref=True):
    ''' Loads eeg data from specified patient and returns an xarray
    Inputs: 
    Patient_numb: string example '/SL01'
    frequency: integer in s^-1
    drop_ref: Drops grounding electrodes and refence electrodes if true
    Outputs:
    eeg_data: xarray
    '''
    # Load the data
    dropped_electrodes=['FCz', 'AFz', 'Nasion']
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
            if drop_ref==True:
                if Name[i].get_text() in dropped_electrodes:
                    pass
                else:
                    rows = [Name[i].get_text(), float(X[i].get_text()), 
                            float(Y[i].get_text()), float(Z[i].get_text()), 
                            float(Theta[i].get_text()), float(Phi[i].get_text()),
                            float(Radius[i].get_text()), float(Channel[i].get_text())]
                    data.append(rows)  
            else:
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
# electrodes = electrodes.values  # Convert DataArray to numpy array
# electrodes=electrodes[0]
# electrodes_2 = electrode_labels().values  # Convert DataArray to numpy array
# Strip spaces from electrode labels
# electrodes_2 = np.array([ele.strip() for ele in electrodes_2])
# print(electrodes.shape, electrodes_2.shape)
# drop_electrodes = np.array([])
# for electrode in electrodes:
    # if electrode not in electrodes_2:
        # drop_electrodes = np.append(drop_electrodes, electrode)

# print(drop_electrodes)
# one_sel=patient_spatial.sel(Values=1).astype(float).to_numpy()
# print(one_sel[0][:])


    

