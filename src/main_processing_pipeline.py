# src/main_processing_pipeline.py
# Main script to process the SisFall dataset: load, preprocess, extract features.

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import Tuple

import config
import data_loader
import preprocessing
import feature_extraction

def process_full_dataset() -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Processes the SisFall dataset for ALL found subjects and ALL their respective activities.
    - Loads data for all subjects and all their activities.
    - Applies preprocessing (filtering, segmentation).
    - Extracts features from each window.
    - Compiles all features and labels.

    Returns:
        Tuple[pd.DataFrame, np.ndarray]:
            - DataFrame of all extracted features (rows are windows, columns are features).
            - NumPy array of corresponding labels for each window.
    """
    all_features_list = []
    all_labels_list = []

    subject_ids_to_process = data_loader.get_all_subject_ids()
    
    if not subject_ids_to_process:
        print("No subject IDs found by data_loader.get_all_subject_ids(). Please check your dataset path and structure.")
        return pd.DataFrame(), np.array([])

    print(f"Starting full dataset processing for {len(subject_ids_to_process)} subjects and all their activities...")

    for subject_id in tqdm(subject_ids_to_process, desc="Processing Subjects"):
        activities_map = data_loader.get_all_activity_codes_for_subject(subject_id)
        
        if not activities_map:
            print(f"INFO: No activities found for subject {subject_id}. Skipping this subject.")
            continue

        for activity_code, trial_files in activities_map.items():
            if not trial_files:
                print(f"INFO: No trial files found for activity {activity_code} of subject {subject_id}. Skipping this activity.")
                continue
            
            label = 1 if activity_code.upper().startswith('F') else 0

            for trial_file_path in trial_files:
                # 1. Load data
                raw_sensor_df = data_loader.load_sisall_file(trial_file_path)
                if raw_sensor_df is None or raw_sensor_df.empty:
                    continue

                # 2. Preprocessing: Filter
                if len(raw_sensor_df) < config.WINDOW_SAMPLES: 
                    filter_coeff_len = config.BUTTERWORTH_ORDER + 1 
                    if len(raw_sensor_df) <= 3 * filter_coeff_len : 
                        print(f"Skipping file {trial_file_path} due to insufficient length for filtering: {len(raw_sensor_df)} samples (need > {3 * filter_coeff_len}).")
                        continue
                
                filtered_sensor_df = preprocessing.apply_butterworth_lowpass_filter(raw_sensor_df)
                
                if filtered_sensor_df.empty:
                    print(f"INFO: DataFrame empty after filtering for {trial_file_path}. Original length: {len(raw_sensor_df)}.")
                    continue
                
                # 3. Preprocessing: Segment into windows
                windows = preprocessing.segment_data(
                    filtered_sensor_df,
                    window_samples=config.WINDOW_SAMPLES,
                    step_samples=config.WINDOW_STEP_SAMPLES
                )

                if not windows:
                    continue
                    
                # 4. Feature Extraction for each window
                for window_df in windows:
                    if window_df.shape[0] == config.WINDOW_SAMPLES:
                        feature_series = feature_extraction.extract_features_from_window(window_df)
                        all_features_list.append(feature_series)
                        all_labels_list.append(label)

    if not all_features_list:
        print("CRITICAL: No features were extracted from the entire dataset. Check pipeline thoroughly.")
        return pd.DataFrame(), np.array([])

    features_df = pd.concat(all_features_list, axis=1).T
    features_df.reset_index(drop=True, inplace=True)
    labels_np = np.array(all_labels_list)

    return features_df, labels_np


if __name__ == "__main__":
    print("--- Main Processing Pipeline (Full Dataset) ---")
    
    features_dataframe, labels_array = process_full_dataset()

    if not features_dataframe.empty:
        print("\nFull Dataset Processing Complete.")
        print(f"Shape of final features DataFrame: {features_dataframe.shape}")
        print(f"Shape of final labels array: {labels_array.shape}")
        
        features_savename = f"sisall_features_ALL_SUBJECTS_ALL_ACTIVITIES_{config.FEATURE_SET_NAME}_w{config.WINDOW_SECONDS}s_o{int(config.WINDOW_OVERLAP_PERCENT*100)}.csv"
        labels_savename = f"sisall_labels_ALL_SUBJECTS_ALL_ACTIVITIES_{config.FEATURE_SET_NAME}_w{config.WINDOW_SECONDS}s_o{int(config.WINDOW_OVERLAP_PERCENT*100)}.npy"
        
        features_save_path = os.path.join(config.FEATURES_PATH, features_savename)
        labels_save_path = os.path.join(config.FEATURES_PATH, labels_savename)

        try:
            features_dataframe.to_csv(features_save_path, index=False)
            np.save(labels_save_path, labels_array)
            print(f"Full Processed Features saved to: {features_save_path}")
            print(f"Full Processed Labels saved to: {labels_save_path}")
        except Exception as e:
            print(f"Error saving full processed data: {e}")

        print("\nSample of Full Features DataFrame:")
        print(features_dataframe.head())
        print("\nUnique labels and their counts in the full processed dataset:")
        unique_labels, counts = np.unique(labels_array, return_counts=True)
        for label_val, count in zip(unique_labels, counts):
            print(f"Label {label_val}: {count} samples")
    else:
        print("No data processed from the full dataset attempt. Check logs for errors.")