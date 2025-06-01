# src/main_processing_pipeline.py
# Main script to process the SisFall dataset: load, preprocess, extract features.

import os
import pandas as pd
import numpy as np
from tqdm import tqdm # For progress bars
from typing import List, Union, Optional, Tuple # Ensure these are imported

import config
import data_loader
import preprocessing
import feature_extraction

def process_dataset(target_subject_ids: List[str], target_activity_codes_list: List[str]) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Processes the SisFall dataset for specified subjects and a list of target activities.
    - Loads data for specified subjects and activities.
    - Applies preprocessing (filtering, segmentation).
    - Extracts features from each window.
    - Compiles all features and labels.

    Args:
        target_subject_ids (List[str]): List of subject IDs to process (e.g., ['SA01', 'SA02']).
        target_activity_codes_list (List[str]): List of activity codes to process (e.g., ['D11', 'F01']).

    Returns:
        Tuple[pd.DataFrame, np.ndarray]:
            - DataFrame of all extracted features (rows are windows, columns are features).
            - NumPy array of corresponding labels for each window.
    """
    all_features_list = []
    all_labels_list = []

    print(f"Starting dataset processing for {len(target_subject_ids)} subjects, focusing on activities: {target_activity_codes_list}...")

    for subject_id in tqdm(target_subject_ids, desc="Processing Subjects"):
        activities_map = data_loader.get_all_activity_codes_for_subject(subject_id)
        
        # --- CORRECTED LOGIC FOR TARGET_ACTIVITY_CODES ---
        for current_activity_code in target_activity_codes_list: # Iterate through the list of target codes
            if current_activity_code not in activities_map:
                print(f"INFO: Activity {current_activity_code} not found for subject {subject_id}. Skipping this activity for this subject.")
                continue 

            trial_files = activities_map[current_activity_code] # Use the individual activity code as the key
            
            # Determine label: 1 if activity code starts with 'F' (Fall), else 0 (ADL)
            label = 1 if current_activity_code.upper().startswith('F') else 0
            # --- END OF CORRECTION ---

            for trial_file_path in trial_files:
                # 1. Load data
                raw_sensor_df = data_loader.load_sisall_file(trial_file_path)
                if raw_sensor_df is None or raw_sensor_df.empty:
                    print(f"Skipping empty or unreadable file: {trial_file_path}")
                    continue

                # 2. Preprocessing: Filter
                if len(raw_sensor_df) < config.WINDOW_SAMPLES: 
                    filter_coeff_len = config.BUTTERWORTH_ORDER + 1 
                    if len(raw_sensor_df) <= 3 * filter_coeff_len :
                        print(f"Skipping file {trial_file_path} due to insufficient length for filtering: {len(raw_sensor_df)} samples.")
                        continue
                
                filtered_sensor_df = preprocessing.apply_butterworth_lowpass_filter(raw_sensor_df)
                
                if filtered_sensor_df.empty:
                    print(f"Skipping file {trial_file_path} as it became empty after filtering.")
                    continue
                
                # 3. Preprocessing: Segment into windows
                windows = preprocessing.segment_data(
                    filtered_sensor_df,
                    window_samples=config.WINDOW_SAMPLES,
                    step_samples=config.WINDOW_STEP_SAMPLES
                )

                if not windows:
                    print(f"No windows generated for {trial_file_path}. Length after filtering: {len(filtered_sensor_df)}. Window size: {config.WINDOW_SAMPLES}")
                    continue
                    
                # 4. Feature Extraction for each window
                for window_df in windows:
                    if window_df.shape[0] == config.WINDOW_SAMPLES: 
                        feature_series = feature_extraction.extract_features_from_window(window_df)
                        all_features_list.append(feature_series)
                        all_labels_list.append(label) # Label is now correctly set for current_activity_code
                    else:
                        print(f"Warning: Window from {trial_file_path} has incorrect shape {window_df.shape}. Expected {config.WINDOW_SAMPLES} samples. Skipping window.")

    if not all_features_list:
        print(f"No features were extracted for the specified subjects/activities. Check data paths, loading, and processing steps.")
        return pd.DataFrame(), np.array([])

    features_df = pd.concat(all_features_list, axis=1).T 
    features_df.reset_index(drop=True, inplace=True)
    labels_np = np.array(all_labels_list)

    return features_df, labels_np


if __name__ == "__main__":
    # --- Configuration for this test run ---
    # You had subject_ids_to_process = data_loader.get_all_subject_ids() in your snippet
    # For testing, let's keep it limited.
    SUBJECTS_TO_RUN = ['SA01'] # Process only subject SA01 for a quick test
    # SUBJECTS_TO_RUN = data_loader.get_all_subject_ids() # Uncomment to process all subjects
    
    ACTIVITIES_TO_RUN = ["D11", "F01"] # An ADL and a Fall
    
    print(f"--- Main Processing Pipeline Test (Focusing on Subjects: {SUBJECTS_TO_RUN}, Activities: {ACTIVITIES_TO_RUN}) ---")
    
    # Ensure the function signature change in process_dataset is reflected in the call:
    features_dataframe, labels_array = process_dataset(
        target_subject_ids=SUBJECTS_TO_RUN,
        target_activity_codes_list=ACTIVITIES_TO_RUN # Pass the list here
    )

    if not features_dataframe.empty:
        print(f"\nProcessing Complete for selected activities.")
        print(f"Shape of final features DataFrame: {features_dataframe.shape}")
        print(f"Shape of final labels array: {labels_array.shape}")
        
        subject_tag = "_".join(SUBJECTS_TO_RUN) if len(SUBJECTS_TO_RUN) < 4 else f"{len(SUBJECTS_TO_RUN)}subjects"
        activity_tag = "_".join(ACTIVITIES_TO_RUN) if len(ACTIVITIES_TO_RUN) < 4 else f"{len(ACTIVITIES_TO_RUN)}activities"
        
        features_savename = f"sisall_features_{subject_tag}_{activity_tag}_{config.FEATURE_SET_NAME}_w{config.WINDOW_SECONDS}s_o{int(config.WINDOW_OVERLAP_PERCENT*100)}.csv"
        labels_savename = f"sisall_labels_{subject_tag}_{activity_tag}_{config.FEATURE_SET_NAME}_w{config.WINDOW_SECONDS}s_o{int(config.WINDOW_OVERLAP_PERCENT*100)}.npy"
        
        features_save_path = os.path.join(config.FEATURES_PATH, features_savename)
        labels_save_path = os.path.join(config.FEATURES_PATH, labels_savename)

        try:
            features_dataframe.to_csv(features_save_path, index=False)
            np.save(labels_save_path, labels_array)
            print(f"Subset Features saved to: {features_save_path}")
            print(f"Subset Labels saved to: {labels_save_path}")
        except Exception as e:
            print(f"Error saving subset processed data: {e}")

        print("\nSample of Subset Features DataFrame:")
        print(features_dataframe.head())
        print("\nUnique labels and their counts in this subset:")
        unique_labels, counts = np.unique(labels_array, return_counts=True)
        for label_val, count in zip(unique_labels, counts):
            print(f"Label {label_val}: {count} samples") # Should show counts for both 0 and 1
    else:
        print(f"No data processed for Subjects: {SUBJECTS_TO_RUN}, Activities: {ACTIVITIES_TO_RUN}. Check logs for errors.")