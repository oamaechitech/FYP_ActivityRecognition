# src/preprocessing.py
# Functions for preprocessing wearable sensor data.

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, lfilter # For Butterworth filter
from typing import List, Union, Optional, Tuple
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import List, Union, Optional

import config # Assuming config.py is in the same directory or accessible

def apply_butterworth_lowpass_filter(data: Union[pd.DataFrame, np.ndarray],
                                     cutoff_hz: float = config.BUTTERWORTH_CUTOFF_HZ,
                                     sampling_rate_hz: float = config.SAMPLING_RATE_HZ,
                                     order: int = config.BUTTERWORTH_ORDER) -> Union[pd.DataFrame, np.ndarray]:
    """
    Applies a Butterworth low-pass filter to the sensor data.
    Consistent with Sucerquia et al. (2017) initial analysis of SisFall.

    Args:
        data (Union[pd.DataFrame, np.ndarray]): Input sensor data. If DataFrame, assumes columns are sensor channels.
                                                If NumPy array, assumes shape (n_samples, n_channels).
        cutoff_hz (float): The cutoff frequency of the filter in Hz.
        sampling_rate_hz (float): The sampling rate of the sensor data in Hz.
        order (int): The order of the Butterworth filter.

    Returns:
        Union[pd.DataFrame, np.ndarray]: Filtered data in the same format as input.
    """
    nyquist_freq_hz = 0.5 * sampling_rate_hz
    normal_cutoff = cutoff_hz / nyquist_freq_hz
    
    # Get the filter coefficients
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    if isinstance(data, pd.DataFrame):
        filtered_data = data.copy()
        for column in filtered_data.columns:
            # Using filtfilt for zero phase distortion
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
                                Assumes rows are samples and columns are sensor channels.
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
    Zurbuchen et al. (2021) used min-max normalization for features.

    Args:
        features_train (np.ndarray): Training feature array (n_samples, n_features).
        features_test (Optional[np.ndarray]): Test feature array (n_samples, n_features). Can be None.
        method (str): Normalization method ('minmax' or 'standard').

    Returns:
        If features_test is provided:
            Tuple[np.ndarray, np.ndarray, object]: Normalized train features, normalized test features, fitted scaler.
        If features_test is None:
            Tuple[np.ndarray, object]: Normalized train features, fitted scaler.
    """
    if method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'standard':
        scaler = StandardScaler()
    else:
        raise ValueError("Method must be 'minmax' or 'standard'")

    # Fit on training data and transform it
    features_train_scaled = scaler.fit_transform(features_train)

    if features_test is not None:
        # Transform test data using the scaler fitted on training data
        features_test_scaled = scaler.transform(features_test)
        return features_train_scaled, features_test_scaled, scaler
    else:
        return features_train_scaled, scaler


if __name__ == "__main__":
    print("--- Preprocessing Test ---")
    
    # Create some dummy sensor data (e.g., 5 seconds at 200Hz, 6 channels)
    fs = 200  # Sampling rate
    duration = 5  # seconds
    n_samples_dummy = fs * duration
    n_channels_dummy = 6
    
    # Simulate some sinusoidal data with noise
    time_vec = np.linspace(0, duration, n_samples_dummy, endpoint=False)
    dummy_data_np = np.zeros((n_samples_dummy, n_channels_dummy))
    for i in range(n_channels_dummy):
        signal = np.sin(2 * np.pi * (i + 1) * time_vec) + \
                 0.5 * np.sin(2 * np.pi * (i + 1) * 5 * time_vec) # Base signal
        noise = 0.2 * np.random.randn(n_samples_dummy) # Add some noise
        dummy_data_np[:, i] = signal + noise
    
    dummy_df = pd.DataFrame(dummy_data_np, columns=[f'ch{j+1}' for j in range(n_channels_dummy)])
    print(f"Original dummy data shape: {dummy_df.shape}")

    # 1. Test Butterworth Filter
    print("\nTesting Butterworth filter...")
    filtered_df = apply_butterworth_lowpass_filter(dummy_df, sampling_rate_hz=fs)
    print(f"Filtered dummy data shape: {filtered_df.shape}")
    
    # Optional: Plot to see the effect (requires matplotlib)
    # import matplotlib.pyplot as plt
    # plt.figure(figsize=(12, 6))
    # plt.plot(time_vec, dummy_df['ch1'], label='Original Channel 1')
    # plt.plot(time_vec, filtered_df['ch1'], label='Filtered Channel 1')
    # plt.title("Butterworth Filter Test")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Amplitude")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    # 2. Test Segmentation
    print("\nTesting segmentation...")
    # Use config for window/step, but adjust for dummy data duration if needed
    dummy_window_samples = int(fs * 2.0) # 2-second windows for dummy
    dummy_step_samples = int(dummy_window_samples * 0.5) # 50% overlap
    
    windows = segment_data(filtered_df, window_samples=dummy_window_samples, step_samples=dummy_step_samples)
    if windows:
        print(f"Number of windows created: {len(windows)}")
        print(f"Shape of the first window: {windows[0].shape}")
    else:
        print("No windows created. Check data length and window/step parameters.")

    # 3. Test Normalization (using dummy features)
    print("\nTesting feature normalization...")
    # Create dummy features (e.g., from the windows)
    if windows:
        # Simple mean features for testing normalization
        dummy_features_list = [win.mean().values for win in windows] # .values to get numpy array
        dummy_features_np = np.array(dummy_features_list)
        
        if dummy_features_np.ndim == 1: # If only one window was created
             dummy_features_np = dummy_features_np.reshape(1, -1)

        if dummy_features_np.shape[0] > 1: # Need at least 2 samples for train/test split
            # Split into dummy train/test
            split_idx = int(0.7 * len(dummy_features_np))
            dummy_features_train = dummy_features_np[:split_idx]
            dummy_features_test = dummy_features_np[split_idx:]

            print(f"Dummy features train shape: {dummy_features_train.shape}")
            print(f"Dummy features test shape: {dummy_features_test.shape}")

            # Test MinMax scaling
            norm_train_minmax, norm_test_minmax, scaler_minmax = normalize_features(
                dummy_features_train, dummy_features_test, method='minmax'
            )
            print("MinMax Scaled Train Features (first 2 samples, first 3 features):")
            print(norm_train_minmax[:2, :3])
            print("MinMax Scaled Test Features (first 2 samples, first 3 features):")
            if len(norm_test_minmax) > 0: print(norm_test_minmax[:2, :3])
            else: print("Test set empty for minmax print.")


            # Test Standard scaling
            norm_train_std, norm_test_std, scaler_std = normalize_features(
                dummy_features_train, dummy_features_test, method='standard'
            )
            print("Standard Scaled Train Features (first 2 samples, first 3 features):")
            print(norm_train_std[:2, :3])
            print("Standard Scaled Test Features (first 2 samples, first 3 features):")
            if len(norm_test_std) > 0: print(norm_test_std[:2, :3])
            else: print("Test set empty for standard print.")
        else:
            print("Not enough dummy feature samples to test train/test normalization split.")
            if dummy_features_np.shape[0] > 0 :
                norm_train_minmax_single, scaler_minmax_single = normalize_features(dummy_features_np, method='minmax')
                print("MinMax Scaled Single Set (first sample, first 3 features):")
                print(norm_train_minmax_single[0, :3])