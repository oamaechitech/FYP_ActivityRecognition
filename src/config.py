# src/config.py
# Configuration file for the HAR Final Year Project

import os

# --- Project Root ---
# Assuming this config.py is in 'src/', the project root is one level up.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
# Path to the directory where the SisFall dataset is stored/organized
# IMPORTANT: You will need to download SisFall and organize it.
# This path assumes a 'data/SisFall_Dataset' structure within your project root.
# Adjust if your SisFall data is structured differently or located elsewhere.
SISFALL_DATASET_PATH = r'C:\Users\Obinn\Documents\UNI\!Final Year\FINAL PROJECT\FYP_ActivityRecognition\data\SisFall_dataset' # <<<<------ UPDATED LINE
# Example: If subject SA01's data for activity D01 is in 'SA01/D01_SA01_R01.txt'
# You might access it via os.path.join(SISFALL_DATASET_PATH, "SA01", "D01_SA01_R01.txt")

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
# As per SisFall and Zurbuchen et al. (2021)
# SisFall was recorded at 200 Hz
# Zurbuchen et al. used data from one accelerometer (ADXL345) and the gyroscope (ITG3200)
# ADXL345 sensor range: ±16g
# ITG3200 gyroscope range: ±2000°/s
SAMPLING_RATE_HZ = 200.0

# Columns for ADXL345 Accelerometer and ITG3200 Gyroscope based on typical SisFall usage
# Adjust these based on the actual column names in your SisFall files if they differ.
# Assuming: Ax, Ay, Az for ADXL345; Gx, Gy, Gz for ITG3200
# IMPORTANT: Verify these with the actual SisFall data file format.
# Sucerquia et al. (2017) mention ADXL345 and MMA8451Q accelerometers, and ITG3200 gyroscope.
# Zurbuchen et al. (2021) state they used ADXL345 and ITG3200.
# We'll assume 6 channels of interest as per Zurbuchen et al.
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
# This is just a placeholder; you'll define more specific features in feature_extraction.py
FEATURE_SET_NAME = "initial_stats_zurbuchen_inspired"

# --- Model Training & Evaluation ---
RANDOM_SEED = 42 # For reproducibility
TRAIN_TEST_SPLIT_RATIO = 0.2 # Proportion of data for the test set
N_FOLDS_CROSS_VALIDATION = 5 # For k-fold cross-validation

# Activity labels mapping (example, expand based on SisFall documentation)
# The actual mapping might be more complex (e.g., D01, F01 codes)
ACTIVITY_LABEL_MAP = {
    "D01": "Walking_Slow",
    "D02": "Walking_Quick",
    # ... add all ADL labels
    "F01": "Fall_Forward_Slip",
    # ... add all Fall labels
    # It might be simpler to map these to broader categories like "ADL" and "Fall" initially
}
TARGET_COLUMN_NAME = "activity_label" # The column name for your activity labels

# --- For Zurbuchen et al. (2021) Feature Set ---
# Time-domain features: variance, std_dev, mean, median, max, min, delta (peak-to-peak), 25th_centile, 75th_centile
# Frequency-domain features: power_spectral_density, power_spectral_entropy
# Axes: 3 accel, 3 gyro, 1 accel_mag, 1 gyro_mag = 8 axes
# Total features: 11 feature_classes * 8 axes = 88 features

# You can add more configurations as your project evolves.
print("Config loaded.")
if __name__ == "__main__":
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"SisFall Data Path: {SISFALL_DATASET_PATH}") # This will print your new path
    print(f"Window Samples: {WINDOW_SAMPLES}")