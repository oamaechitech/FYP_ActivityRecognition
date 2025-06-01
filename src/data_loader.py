# src/data_loader.py
# Functions for loading and interacting with the SisFall dataset.

import os
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional

# Assuming config.py is in the same directory or accessible in PYTHONPATH
import config

def get_subject_activity_files(subject_id: str, activity_id: str) -> List[str]:
    """
    Finds all data files for a given subject and activity.
    SisFall Naming: D01_SA01_R01.txt (Activity_Subject_Trial.txt)
                    F01_SA01_R01.txt
    Args:
        subject_id (str): Subject identifier (e.g., "SA01", "SE01").
        activity_id (str): Activity identifier (e.g., "D01", "F01").
    Returns:
        List[str]: List of full file paths for the given subject and activity.
    """
    subject_files = []
    subject_folder_path = os.path.join(config.SISFALL_DATASET_PATH, subject_id)

    if not os.path.isdir(subject_folder_path):
        print(f"Warning: Subject folder not found: {subject_folder_path}")
        return []

    for filename in os.listdir(subject_folder_path):
        parts = filename.split('_')
        if len(parts) == 3 and parts[0] == activity_id and parts[1] == subject_id:
            if filename.endswith(".txt"):
                file_path = os.path.join(subject_folder_path, filename)
                subject_files.append(file_path)
    return sorted(subject_files)


def load_sisall_file(file_path: str) -> Optional[pd.DataFrame]:
    """
    Loads a single SisFall data file into a pandas DataFrame.
    The SisFall files are comma-separated, with a trailing semicolon on each line.
    Expected 9 columns of sensor readings.
    Zurbuchen et al. (2021) used ADXL345 (one accelerometer) and ITG3200 (gyroscope).

    Args:
        file_path (str): The full path to the SisFall data file.
    Returns:
        Optional[pd.DataFrame]: DataFrame with selected sensor readings, or None if loading fails.
    """
    print(f"INFO: Attempting to load: {file_path}")
    try:
        col_names_raw_ordered = [
            'raw_acc1_x', 'raw_acc1_y', 'raw_acc1_z',
            'raw_gyro_x', 'raw_gyro_y', 'raw_gyro_z',
            'raw_acc2_x', 'raw_acc2_y', 'raw_acc2_z'
        ]

        df = pd.read_csv(file_path, header=None, sep=',')

        if df.empty:
            print(f"Warning: DataFrame is empty immediately after read_csv for {file_path}")
            return None

        if df.shape[1] == len(col_names_raw_ordered):
            if df.iloc[:, -1].dtype == 'object':
                df.iloc[:, -1] = df.iloc[:, -1].astype(str).str.rstrip(';')
            df.columns = col_names_raw_ordered
        elif df.shape[1] == 1 and isinstance(df.iloc[0,0], str) and df.iloc[0,0].count(';') > 0 and df.iloc[0,0].count(';') == df.iloc[0,0].count(','):
            print(f"INFO: Possibly all data in one column for {file_path}. Trying to split again.")
            data_lines = []
            with open(file_path, 'r') as f:
                for line in f:
                    cleaned_line = line.strip().rstrip(';')
                    if cleaned_line:
                        values = [val.strip() for val in cleaned_line.split(',')]
                        if len(values) == len(col_names_raw_ordered):
                            data_lines.append(values)
            if not data_lines:
                print(f"Warning: No valid data lines found after manual parsing for {file_path}")
                return None
            df = pd.DataFrame(data_lines, columns=col_names_raw_ordered)
        else:
            print(f"Warning: Expected {len(col_names_raw_ordered)} columns but found {df.shape[1]} in {file_path}. Content head:\n{df.head()}")
            if df.shape[1] >= len(col_names_raw_ordered):
                 df.columns = col_names_raw_ordered[:df.shape[1]]
            else:
                print(f"ERROR: Insufficient columns in {file_path} to proceed.")
                return None

        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        column_map = {
            'raw_acc1_x': 'acc_x', 'raw_acc1_y': 'acc_y', 'raw_acc1_z': 'acc_z',
            'raw_gyro_x': 'gyro_x', 'raw_gyro_y': 'gyro_y', 'raw_gyro_z': 'gyro_z'
        }
        
        required_cols_for_selection = [col for col in column_map.keys() if col in df.columns]
        if len(required_cols_for_selection) != len(column_map):
            print(f"Warning: Not all expected raw columns for mapping are present in {file_path} after numeric conversion.")
            print(f"  Expected for mapping: {list(column_map.keys())}")
            print(f"  Found in DataFrame: {df.columns.tolist()}")
            return None

        df_selected = df[required_cols_for_selection].rename(columns=column_map)
        
        if not all(col in df_selected.columns for col in config.SENSOR_COLUMNS):
            print(f"ERROR: Not all target SENSOR_COLUMNS are present in df_selected for {file_path} after mapping.")
            print(f"  Target SENSOR_COLUMNS: {config.SENSOR_COLUMNS}")
            print(f"  Columns in df_selected: {df_selected.columns.tolist()}")
            return None
        df_selected = df_selected[config.SENSOR_COLUMNS]

        rows_before_dropna = len(df_selected)
        df_selected.dropna(inplace=True)
        df_selected.reset_index(drop=True, inplace=True)
        rows_after_dropna = len(df_selected)

        if rows_before_dropna > 0 and rows_after_dropna == 0:
            print(f"Warning: All rows were dropped by dropna() for {file_path}. Check for widespread NaNs.")
        elif rows_before_dropna > rows_after_dropna :
             print(f"INFO: Dropped {rows_before_dropna - rows_after_dropna} rows with NaNs from {file_path}.")

        if df_selected.empty:
            print(f"Warning: DataFrame is empty after all processing for file: {file_path}")
            return None
            
        return df_selected

    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        import traceback
        traceback.print_exc()
        return None

# --- ADD THESE FUNCTIONS BACK IN ---
def get_all_subject_ids() -> List[str]:
    """
    Scans the SisFall dataset path and returns a list of all subject IDs (folder names).
    Example Subject IDs: SA01, SA02, ..., SE01, SE02, ...
    """
    subject_ids = []
    if not os.path.isdir(config.SISFALL_DATASET_PATH):
        print(f"Error: SisFall dataset path not found: {config.SISFALL_DATASET_PATH}")
        print("Please download the SisFall dataset and update SISFALL_DATASET_PATH in config.py")
        return []
        
    for item_name in os.listdir(config.SISFALL_DATASET_PATH):
        item_path = os.path.join(config.SISFALL_DATASET_PATH, item_name)
        # Assuming subject folders start with 'S' (e.g., SA, SE)
        # And are directories
        if os.path.isdir(item_path) and item_name.upper().startswith('S'): # Use .upper() for case-insensitivity if needed
            subject_ids.append(item_name)
    return sorted(subject_ids)

def get_all_activity_codes_for_subject(subject_id: str) -> Dict[str, List[str]]:
    """
    Gets all activity codes (e.g., 'D01', 'F01') and their trial file paths for a given subject.
    Returns a dictionary where keys are activity codes and values are lists of file paths.
    """
    activity_files_map = {}
    subject_folder_path = os.path.join(config.SISFALL_DATASET_PATH, subject_id)

    if not os.path.isdir(subject_folder_path):
        print(f"Warning: Subject folder not found: {subject_folder_path}")
        return {}

    for filename in os.listdir(subject_folder_path):
        if filename.endswith(".txt"): # Or other extension
            parts = filename.split('_')
            # Example: D01_SA01_R01.txt
            if len(parts) == 3 and parts[1].upper() == subject_id.upper(): # Compare subject_id case-insensitively
                activity_code = parts[0]
                file_path = os.path.join(subject_folder_path, filename)
                if activity_code not in activity_files_map:
                    activity_files_map[activity_code] = []
                activity_files_map[activity_code].append(file_path)
    
    for code in activity_files_map: # Sort trial files
        activity_files_map[code].sort()
        
    return activity_files_map
# --- END OF ADDED FUNCTIONS ---

if __name__ == "__main__":
    print("--- Data Loader Test ---")
    
    subject_ids = get_all_subject_ids() # Now this function exists
    if not subject_ids:
        print("No subject IDs found. Ensure SISFALL_DATASET_PATH in config.py is correct and dataset is present.")
    else:
        print(f"Found {len(subject_ids)} subjects. First 5: {subject_ids[:5]}")

        if subject_ids:
            test_subject = subject_ids[0] 
            print(f"\nTesting for subject: {test_subject}")
            
            activities_for_subject = get_all_activity_codes_for_subject(test_subject) # Now this function exists
            if not activities_for_subject:
                print(f"No activities found for subject {test_subject}")
            else:
                test_file_path = os.path.join(config.SISFALL_DATASET_PATH, test_subject, "D11_SA01_R02.txt")
                if not os.path.exists(test_file_path):
                     first_activity_code = sorted(list(activities_for_subject.keys()))[0]
                     if activities_for_subject[first_activity_code]: # Check if list is not empty
                        test_file_path = activities_for_subject[first_activity_code][0]
                     else:
                         print(f"No trial files found for activity {first_activity_code} of subject {test_subject}")
                         test_file_path = None # Prevent error if no files
                
                if test_file_path:
                    print(f"Attempting to load file: {test_file_path}")
                    df_sample = load_sisall_file(test_file_path)

                    if df_sample is not None and not df_sample.empty:
                        print(f"\nSuccessfully loaded {test_file_path}:")
                        print(f"Shape: {df_sample.shape}")
                        print("Head:")
                        print(df_sample.head())
                        print("\nInfo:")
                        df_sample.info() 
                        print("\nDescribe:")
                        print(df_sample.describe())
                    else:
                        print(f"Failed to load or resulted in empty DataFrame for {test_file_path}.")
                else:
                    print(f"No suitable test file path determined for subject {test_subject}")