import numpy as np
from scipy.linalg import lstsq

def remove_eye_artifact(eeg, eog, debug=False):
    """
    Removes eye artifacts from EEG data using EOG channels via linear regression.

    Parameters:
    eeg : np.array
        EEG data array of shape (n_samples, 60) where 60 is the number of EEG channels.
    eog : np.array
        EOG data array of shape (n_samples, 4) where 4 is the number of EOG channels.
    debug : bool
        If True, prints debugging information.

    Returns:
    np.array
        EEG data with eye artifacts removed, same shape as input EEG data.
    """

    # Add a column of ones to EOG data for intercept in the regression model
    X = np.hstack([eog, np.ones((eog.shape[0], 1))])
    
    cleaned_eeg = np.zeros_like(eeg)
    # check if the number of channels is 60
    if debug:
        print(f"Number of EEG channels: {eeg.shape[1]}")
    assert eeg.shape[1] == 60, "Number of EEG channels must be 60."
    
    # 60 channels
    for i in range(eeg.shape[1]):
        # Get the current EEG channel data
        y = eeg[:, i]
        
        # Calculate regression coefficients (including intercept)
        coefficients = lstsq(X, y)[0]
        eog_contribution = X.dot(coefficients)
        
        # Subtract the EOG contribution from the original EEG data
        cleaned_eeg[:, i] = y - eog_contribution
    
    return cleaned_eeg