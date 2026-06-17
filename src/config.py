# src/config.py


import os

# --- Project Root ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
# Path to the directory where the SisFall dataset is stored/organized
SISFALL_DATASET_PATH = r'C:\Users\Obinn\Documents\UNI\!Final Year\FINAL PROJECT\FYP_ActivityRecognition\data\SisFall_dataset'


PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
if not os.path.exists(PROCESSED_DATA_PATH):
    os.makedirs(PROCESSED_DATA_PATH)

# --- Feature Engineering & Model Paths ---
FEATURES_PATH = os.path.join(PROJECT_ROOT, "data", "features")
if not os.path.exists(FEATURES_PATH):
    os.makedirs(FEATURES_PATH)

MODELS_PATH = os.path.join(PROJECT_ROOT, "models")
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")
if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)

# --- Data Characteristics ---

SAMPLING_RATE_HZ = 200.0



SENSOR_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
N_SENSOR_CHANNELS = len(SENSOR_COLUMNS)

# --- Preprocessing Parameters ---
# Windowing parameters for segmentation (based on Zurbuchen et al. 10s records for features)
# 10 seconds at 200 Hz = 2000 samples
WINDOW_SECONDS = 10.0
WINDOW_SAMPLES = int(SAMPLING_RATE_HZ * WINDOW_SECONDS)
# Overlap: Zurbuchen et al. processed distinct 10s windows after trimming.
# For training, some overlap might be useful. Let's start with 50% for now as an example.
# If you follow Zurbuchen et al. more closely, you'd make this 0 after initial file processing.
WINDOW_OVERLAP_PERCENT = 0.50
WINDOW_OVERLAP_SAMPLES = int(WINDOW_SAMPLES * WINDOW_OVERLAP_PERCENT)
WINDOW_STEP_SAMPLES = WINDOW_SAMPLES - WINDOW_OVERLAP_SAMPLES


# Noise Filtering Parameters (example from Sucerquia et al. (2017) for SisFall)
BUTTERWORTH_ORDER = 4
BUTTERWORTH_CUTOFF_HZ = 5.0 # Hz

# --- Feature Engineering ---
# List of features to extract, can be expanded based on Zurbuchen et al. (2021)

FEATURE_SET_NAME = "initial_stats_zurbuchen_inspired"

# --- Model Training & Evaluation ---
RANDOM_SEED = 42 # For reproducibility
TRAIN_TEST_SPLIT_RATIO = 0.2 # Proportion of data for the test set
N_FOLDS_CROSS_VALIDATION = 5 # For k-fold cross-validation

# Activity labels mapping 
ACTIVITY_LABEL_MAP = {
    "D01": "Walking_Slow",
    "D02": "Walking_Quick",
  
    "F01": "Fall_Forward_Slip",
    
}
TARGET_COLUMN_NAME = "activity_label" 




print("Config loaded.")
if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"SisFall Data Path: {SISFALL_DATASET_PATH}") 
    print(f"Window Samples: {WINDOW_SAMPLES}")