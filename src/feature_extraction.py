# src/feature_extraction.py
# Functions for extracting features from windowed sensor data.

import numpy as np
import pandas as pd
from scipy.fft import fft
from scipy.stats import skew, kurtosis, entropy
from typing import List, Dict

import config 

# --- Helper for Magnitude ---
def calculate_magnitude(df_window: pd.DataFrame, acc_cols: List[str], gyro_cols: List[str]) -> pd.DataFrame:
    """
    Calculates accelerometer and gyroscope magnitudes and adds them as new columns.
    Matches Zurbuchen et al. (2021) approach of using magnitudes.
    Args:
        df_window (pd.DataFrame): Window of sensor data with accelerometer and gyroscope columns.
        acc_cols (List[str]): List of accelerometer column names (e.g., ['acc_x', 'acc_y', 'acc_z']).
        gyro_cols (List[str]): List of gyroscope column names (e.g., ['gyro_x', 'gyro_y', 'gyro_z']).
    Returns:
        pd.DataFrame: Original DataFrame with 'acc_mag' and 'gyro_mag' columns appended.
    """
    df_out = df_window.copy()
    if len(acc_cols) == 3:
        df_out['acc_mag'] = np.sqrt(df_out[acc_cols[0]]**2 + df_out[acc_cols[1]]**2 + df_out[acc_cols[2]]**2)
    else:
        print("Warning: Accelerometer magnitude requires 3 axis columns.")

    if len(gyro_cols) == 3:
        df_out['gyro_mag'] = np.sqrt(df_out[gyro_cols[0]]**2 + df_out[gyro_cols[1]]**2 + df_out[gyro_cols[2]]**2)
    else:
        print("Warning: Gyroscope magnitude requires 3 axis columns.")
    return df_out

# --- Time-Domain Features (Inspired by Zurbuchen et al. 2021) ---
def get_time_domain_features(series: pd.Series) -> Dict[str, float]:
    """Calculates basic time-domain statistical features for a single sensor channel (series)."""
    features = {}
    features['mean'] = series.mean()
    features['std'] = series.std()
    features['var'] = series.var()
    features['min'] = series.min()
    features['max'] = series.max()
    features['median'] = series.median()
    features['delta'] = series.max() - series.min() # Peak-to-peak
    features['q25'] = series.quantile(0.25) # 25th centile
    features['q75'] = series.quantile(0.75) # 75th centile
    features['iqr'] = features['q75'] - features['q25']
    features['skew'] = skew(series)         
    features['kurt'] = kurtosis(series)     # Kurtosis can also be useful
    return features

# --- Frequency-Domain Features (Inspired by Zurbuchen et al. 2021) ---
def get_frequency_domain_features(series: pd.Series, sampling_rate: float = config.SAMPLING_RATE_HZ) -> Dict[str, float]:
    """Calculates basic frequency-domain features for a single sensor channel (series)."""
    features = {}
    n = len(series)
    if n == 0:
        features['psd_mean'] = 0.0 # Or np.nan
        features['spectral_entropy'] = 0.0 # Or np.nan
        return features

    # FFT
    yf = fft(series.values)
    # Power Spectral Density (PSD)
    # Calculate one-sided PSD. The factor of 2 accounts for energy in negative frequencies (except DC and Nyquist)
    psd = (2.0 / (n * sampling_rate)) * (np.abs(yf[0:n//2])**2) # Using sampling_rate for proper density scaling
    
    if len(psd) == 0: # Handle cases where series might be too short for n//2
        features['psd_mean'] = 0.0
        features['spectral_entropy'] = 0.0
        return features

    features['psd_mean'] = np.mean(psd) if len(psd) > 0 else 0.0

    # Spectral Entropy
    # Normalize PSD to be like a probability distribution for entropy calculation
    psd_normalized = psd / np.sum(psd) if np.sum(psd) > 0 else psd
    features['spectral_entropy'] = entropy(psd_normalized) if len(psd_normalized) > 0 else 0.0
    
    return features

# --- Main Feature Extraction Function ---
def extract_features_from_window(df_window: pd.DataFrame) -> pd.Series:
    """
    Extracts a comprehensive set of features from a single window of sensor data.
    This function will iterate over specified sensor channels (and magnitudes)
    and calculate both time and frequency domain features.

    Args:
        df_window (pd.DataFrame): A window of sensor data.

    Returns:
        pd.Series: A series containing all extracted features for that window,
                   with clearly named indices (e.g., 'acc_x_mean', 'gyro_z_psd_mean').
    """
    all_features = {}

    # Define accelerometer and gyroscope columns for magnitude calculation
    acc_cols = [col for col in ['acc_x', 'acc_y', 'acc_z'] if col in df_window.columns]
    gyro_cols = [col for col in ['gyro_x', 'gyro_y', 'gyro_z'] if col in df_window.columns]

    # Calculate magnitudes
    df_window_with_mag = calculate_magnitude(df_window, acc_cols, gyro_cols)

    # Define channels to extract features from (original axes + magnitudes)
    # Zurbuchen et al. used 3 accel, 3 gyro, 1 accel_mag, 1 gyro_mag = 8 axes
    channels_to_process = config.SENSOR_COLUMNS[:] # Make a copy
    if 'acc_mag' in df_window_with_mag.columns: channels_to_process.append('acc_mag')
    if 'gyro_mag' in df_window_with_mag.columns: channels_to_process.append('gyro_mag')

    for channel_name in channels_to_process:
        if channel_name not in df_window_with_mag.columns:
            print(f"Warning: Channel '{channel_name}' not found in window for feature extraction. Skipping.")
            continue
            
        series = df_window_with_mag[channel_name]

        # Time-domain features
        time_feats = get_time_domain_features(series)
        for feat_name, feat_val in time_feats.items():
            all_features[f"{channel_name}_{feat_name}"] = feat_val

        # Frequency-domain features
        freq_feats = get_frequency_domain_features(series, sampling_rate=config.SAMPLING_RATE_HZ)
        for feat_name, feat_val in freq_feats.items():
            all_features[f"{channel_name}_{feat_name}"] = feat_val
            
    return pd.Series(all_features)


if __name__ == "__main__":
    print("--- Feature Extraction Test ---")
    
    # Create a dummy window of sensor data (e.g., 10 seconds at 200Hz)
    fs_test = config.SAMPLING_RATE_HZ
    window_len_samples_test = config.WINDOW_SAMPLES # 2000 samples
    
    dummy_window_data = np.random.rand(window_len_samples_test, len(config.SENSOR_COLUMNS))
    dummy_window_df = pd.DataFrame(dummy_window_data, columns=config.SENSOR_COLUMNS)
    print(f"Dummy window shape: {dummy_window_df.shape}")

    # Test feature extraction
    extracted_feature_series = extract_features_from_window(dummy_window_df)
    
    print(f"\nExtracted {len(extracted_feature_series)} features:")
    print("Sample of extracted features:")
    print(extracted_feature_series.head(15)) # Print first 15 features
    
    # Expected number of features (based on Zurbuchen et al. inspiration):
    # Time features: mean, std, var, min, max, median, delta, q25, q75, iqr, skew, kurt (12 features)
    # Freq features: psd_mean, spectral_entropy (2 features)
    # Total per channel: 12 + 2 = 14 features
    # Number of channels (axes): 6 original + 2 magnitudes = 8
    # Expected total: 14 features/channel * 8 channels = 112 features
    print(f"\nTotal number of features extracted: {len(extracted_feature_series)}")
    print("Expected number of features with current setup (approx 12 time + 2 freq per channel): ",
          (len(get_time_domain_features(pd.Series([1,2,3]))) + \
           len(get_frequency_domain_features(pd.Series([1,2,3])))) * \
          (len(config.SENSOR_COLUMNS) + 2) # +2 for magnitudes
         )

    # Test with a short series for frequency features edge case
    print("\nTesting frequency features with very short series:")
    short_series = pd.Series(np.random.rand(1)) # Only 1 sample
    freq_feats_short = get_frequency_domain_features(short_series)
    print(f"Freq features for 1-sample series: {freq_feats_short}")

    empty_series = pd.Series([], dtype=float)
    freq_feats_empty = get_frequency_domain_features(empty_series)
    print(f"Freq features for empty series: {freq_feats_empty}")