# src/preprocessing.py
# Functions for preprocessing wearable sensor data.

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from typing import List, Union, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import config

def apply_butterworth_lowpass_filter(data: Union[pd.DataFrame, np.ndarray],
                                     cutoff_hz: float = config.BUTTERWORTH_CUTOFF_HZ,
                                     sampling_rate_hz: float = config.SAMPLING_RATE_HZ,
                                     order: int = config.BUTTERWORTH_ORDER) -> Union[pd.DataFrame, np.ndarray]:
    """
    Applies a Butterworth low-pass filter to the sensor data.

    Args:
        data (Union[pd.DataFrame, np.ndarray]): Input sensor data.
        cutoff_hz (float): The cutoff frequency of the filter in Hz.
        sampling_rate_hz (float): The sampling rate of the sensor data in Hz.
        order (int): The order of the Butterworth filter.

    Returns:
        Union[pd.DataFrame, np.ndarray]: Filtered data in the same format as input.
    """
    nyquist_freq_hz = 0.5 * sampling_rate_hz
    normal_cutoff = cutoff_hz / nyquist_freq_hz
    
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    if isinstance(data, pd.DataFrame):
        filtered_data = data.copy()
        for column in filtered_data.columns:
            filtered_data[column] = filtfilt(b, a, filtered_data[column])
        return filtered_data
    elif isinstance(data, np.ndarray):
        filtered_data = np.zeros_like(data)
        for i in range(data.shape[1]): # Iterate over channels
            filtered_data[:, i] = filtfilt(b, a, data[:, i])
        return filtered_data
    else:
        raise TypeError("Input data must be a pandas DataFrame or a NumPy ndarray.")

def segment_data(data_df: pd.DataFrame,
                 window_samples: int = config.WINDOW_SAMPLES,
                 step_samples: int = config.WINDOW_STEP_SAMPLES) -> List[pd.DataFrame]:
    """
    Segments the input DataFrame into windows of specified size and step.

    Args:
        data_df (pd.DataFrame): DataFrame containing time-series sensor data.
        window_samples (int): The number of samples in each window.
        step_samples (int): The number of samples to slide the window by.

    Returns:
        List[pd.DataFrame]: A list of DataFrames, where each DataFrame is a window.
    """
    windows = []
    n_samples = len(data_df)
    for i in range(0, n_samples - window_samples + 1, step_samples):
        window = data_df.iloc[i : i + window_samples]
        windows.append(window)
    return windows

def normalize_features(features_train: np.ndarray,
                       features_test: Optional[np.ndarray] = None,
                       method: str = 'minmax') -> Union[Tuple[np.ndarray, Optional[np.ndarray], object],
                                                      Tuple[np.ndarray, object]]:
    """
    Normalizes feature arrays. Fits scaler on training data and applies to both.

    Args:
        features_train (np.ndarray): Training feature array (n_samples, n_features).
        features_test (Optional[np.ndarray]): Test feature array. Can be None.
        method (str): Normalization method ('minmax' or 'standard').

    Returns:
        If features_test is provided:
            Tuple[np.ndarray, np.ndarray, object]: Normalized train features, test features, and scaler.
        If features_test is None:
            Tuple[np.ndarray, object]: Normalized train features and the fitted scaler.
    """
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Method must be 'minmax' or 'standard'")

    features_train_scaled = scaler.fit_transform(features_train)

    if features_test is not None:
        features_test_scaled = scaler.transform(features_test)
        return features_train_scaled, features_test_scaled, scaler
    else:
        return features_train_scaled, scaler