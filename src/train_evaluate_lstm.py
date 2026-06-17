# src/train_evaluate_lstm.py
# Script to define, train, and evaluate an LSTM model for HAR.

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, f1_score
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import config
try:
    import data_loader
    import preprocessing
    from train_evaluate_models import plot_confusion_matrix
except ImportError as e:
    print(f"Error importing utility scripts or plot_confusion_matrix: {e}")
    sys.exit(1)

def prepare_lstm_data_from_trials(max_subjects=None) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """
    Loads raw trial data, preprocesses (filters, segments), and prepares it
    in the sequence format required for LSTM.
    Args:
        max_subjects (int, optional): Max number of subjects to process for debugging.
    Returns:
        X_sequences, y_labels, subject_groups
    """
    all_window_sequences = []
    all_labels = []
    all_subject_group_ids = []

    subject_ids = data_loader.get_all_subject_ids()
    if not subject_ids:
        print("LSTM Data Prep: No subjects found.")
        return None, None, None

    if max_subjects is not None and max_subjects > 0:
        subject_ids = subject_ids[:max_subjects]
        print(f"LSTM Data Prep: Processing a subset of {len(subject_ids)} subjects.")

    print(f"LSTM Data Prep: Starting data preparation for {len(subject_ids)} subjects...")
    
    for subj_idx, subject_id in enumerate(tqdm(subject_ids, desc="LSTM Data Prep - Subjects")):
        activities_map = data_loader.get_all_activity_codes_for_subject(subject_id)
        if not activities_map:
            continue

        for activity_code, trial_files in activities_map.items():
            label = 1 if activity_code.upper().startswith('F') else 0
            for trial_file_path in trial_files:
                raw_sensor_df = data_loader.load_sisall_file(trial_file_path)
                if raw_sensor_df is None or raw_sensor_df.empty:
                    continue
                
                required_len_for_filter = 3 * (config.BUTTERWORTH_ORDER + 1) + 1 
                if len(raw_sensor_df) < required_len_for_filter or len(raw_sensor_df) < config.WINDOW_SAMPLES:
                    continue
                
                filtered_df = preprocessing.apply_butterworth_lowpass_filter(raw_sensor_df)
                if filtered_df is None or filtered_df.empty or len(filtered_df) < config.WINDOW_SAMPLES:
                    continue

                windows_for_trial = preprocessing.segment_data(
                    filtered_df,
                    window_samples=config.WINDOW_SAMPLES,
                    step_samples=config.WINDOW_STEP_SAMPLES
                )

                for window_df in windows_for_trial:
                    if window_df.shape[0] == config.WINDOW_SAMPLES:
                        all_window_sequences.append(window_df.values)
                        all_labels.append(label)
                        all_subject_group_ids.append(subj_idx)

    if not all_window_sequences:
        print("LSTM Data Prep: No sequences generated.")
        return None, None, None

    X_sequences = np.array(all_window_sequences)
    y_labels = np.array(all_labels)
    subject_groups = np.array(all_subject_group_ids)
    
    print(f"LSTM Data Prep: Generated {X_sequences.shape[0]} sequences.")
    print(f"LSTM Data Prep: X_sequences shape: {X_sequences.shape}")
    print(f"LSTM Data Prep: y_labels shape: {y_labels.shape}")
    print(f"LSTM Data Prep: subject_groups shape: {subject_groups.shape}")
    
    return X_sequences, y_labels, subject_groups


def build_lstm_model(input_shape):
    """Defines and compiles the LSTM model architecture."""
    model = Sequential([
        LSTM(64, input_shape=input_shape, return_sequences=True),
        Dropout(0.3),
        BatchNormalization(),
        LSTM(32),
        Dropout(0.3),
        BatchNormalization(),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 
                           tf.keras.metrics.Precision(name='precision'), 
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.AUC(name='auc')])
    return model


if __name__ == "__main__":
    print("--- LSTM Model Training and Evaluation ---")

    MAX_SUBJECTS_FOR_DEBUG = None  

    X_lstm_data, y_lstm_labels, groups_lstm = prepare_lstm_data_from_trials(max_subjects=MAX_SUBJECTS_FOR_DEBUG)

    if X_lstm_data is None or X_lstm_data.size == 0:
        print("Exiting: No data prepared for LSTM training.")
        sys.exit(1)

    if y_lstm_labels.ndim > 1 and y_lstm_labels.shape[1] == 1:
        y_lstm_labels = y_lstm_labels.ravel()

    input_shape_for_lstm = (X_lstm_data.shape[1], X_lstm_data.shape[2])
    print(f"Input shape for LSTM: {input_shape_for_lstm}")

    print(f"\nPerforming Stratified {config.N_FOLDS_CROSS_VALIDATION}-Fold Cross-Validation for LSTM...")
    skf = StratifiedKFold(n_splits=config.N_FOLDS_CROSS_VALIDATION, shuffle=True, random_state=config.RANDOM_SEED)
    
    fold_results = []
    class_names_display_actual = ['ADL (0)', 'Fall (1)'] 
    unique_labels_ordered = np.unique(y_lstm_labels)
    USE_SMOTE_LSTM = False 

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X_lstm_data, y_lstm_labels)):
        print(f"\n--- LSTM - Fold {fold_idx + 1}/{config.N_FOLDS_CROSS_VALIDATION} ---")
        
        X_train_fold, X_test_fold = X_lstm_data[train_index], X_lstm_data[test_index]
        y_train_fold, y_test_fold = y_lstm_labels[train_index], y_lstm_labels[test_index]
        
        # Data Scaling per fold
        scaler_fold = MinMaxScaler(feature_range=(0,1))
        train_shape_orig = X_train_fold.shape
        test_shape_orig = X_test_fold.shape

        if train_shape_orig[-1] == 0:
            print(f"Fold {fold_idx + 1}: No features to scale. Skipping fold.")
            continue 
            
        X_train_fold_reshaped = X_train_fold.reshape(-1, train_shape_orig[-1])
        X_test_fold_reshaped = X_test_fold.reshape(-1, test_shape_orig[-1]) if test_shape_orig[-1] > 0 else np.array([])

        scaler_fold.fit(X_train_fold_reshaped)
        X_train_fold_scaled_reshaped = scaler_fold.transform(X_train_fold_reshaped)
        
        if X_test_fold_reshaped.size > 0:
            X_test_fold_scaled_reshaped = scaler_fold.transform(X_test_fold_reshaped)
        else:
            X_test_fold_scaled_reshaped = np.array([])

        X_train_fold_scaled = X_train_fold_scaled_reshaped.reshape(train_shape_orig)
        if X_test_fold_scaled_reshaped.size > 0:
             X_test_fold_scaled = X_test_fold_scaled_reshaped.reshape(test_shape_orig)
        else:
            X_test_fold_scaled = np.empty((0, test_shape_orig[1], test_shape_orig[2])) if len(test_shape_orig) == 3 else np.empty((0,0))
        
        tf.keras.backend.clear_session()
        lstm_model = build_lstm_model(input_shape=input_shape_for_lstm)

        EPOCHS = 50
        PATIENCE = 10
        early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=PATIENCE // 2, min_lr=0.00001, verbose=1)

        class_weights_dict = None
        if not USE_SMOTE_LSTM: 
            counts = np.bincount(y_train_fold.astype(int))
            if len(counts) == 2 and counts[0] > 0 and counts[1] > 0:
                weight_for_0 = (1 / counts[0]) * (len(y_train_fold)) / 2.0
                weight_for_1 = (1 / counts[1]) * (len(y_train_fold)) / 2.0
                class_weights_dict = {0: weight_for_0, 1: weight_for_1}
                print(f"Using class weights: {class_weights_dict}")

        print(f"Training LSTM on Fold {fold_idx + 1}...")
        validation_data_tuple = None
        if X_test_fold_scaled.shape[0] > 0:
            validation_data_tuple = (X_test_fold_scaled, y_test_fold)

        history = lstm_model.fit(
            X_train_fold_scaled, y_train_fold,
            epochs=EPOCHS, 
            batch_size=64, 
            validation_data=validation_data_tuple, 
            callbacks=[early_stopping, reduce_lr],
            class_weight=class_weights_dict,
            verbose=1 
        )
        
        print(f"Evaluating LSTM on Test Set of Fold {fold_idx + 1}...")
        if X_test_fold_scaled.shape[0] > 0 and y_test_fold.shape[0] > 0:
            loss, acc, prec, rec, auc_val = lstm_model.evaluate(X_test_fold_scaled, y_test_fold, verbose=0)
            
            y_pred_proba_fold = lstm_model.predict(X_test_fold_scaled, verbose=0)
            y_pred_fold = (y_pred_proba_fold > 0.5).astype(int).ravel()

            f1_macro = f1_score(y_test_fold, y_pred_fold, average='macro', zero_division=0)
            f1_fall = f1_score(y_test_fold, y_pred_fold, pos_label=1, average='binary', zero_division=0)
            
            print(classification_report(y_test_fold, y_pred_fold, target_names=class_names_display_actual, labels=unique_labels_ordered, digits=3, zero_division=0))
            report_dict = classification_report(y_test_fold, y_pred_fold, output_dict=True, zero_division=0, labels=unique_labels_ordered, target_names=class_names_display_actual)

            recall_fall_val = report_dict['Fall (1)'].get('recall', 0.0)
            precision_fall_val = report_dict['Fall (1)'].get('precision', 0.0)

            current_metrics = {
                'accuracy': acc, 'precision': prec, 'recall': rec, 'auc': auc_val,
                'f1_macro': f1_macro, 'f1_fall': f1_fall,
                'recall_fall': recall_fall_val,
                'precision_fall': precision_fall_val
            }
            fold_results.append(current_metrics)
            
            print(f"Fold {fold_idx + 1} Results: Acc={acc:.3f}, Prec={prec:.3f}, Rec={rec:.3f}, AUC={auc_val:.3f}, F1Macro={f1_macro:.3f}, F1Fall={f1_fall:.3f}")
            
            if fold_idx == 0: 
                plot_confusion_matrix(y_test_fold, y_pred_fold,
                                      unique_classes_ordered=unique_labels_ordered,
                                      display_classes=class_names_display_actual, 
                                      model_name_tag='LSTM',
                                      fold_num=fold_idx+1,
                                      smote_status_suffix="_SMOTE" if USE_SMOTE_LSTM else "_noSMOTE_CW")
        else:
            print(f"Fold {fold_idx + 1}: Test set was empty. Skipping evaluation.")

    print("\n--- LSTM Cross-Validation Summary ---")
    if fold_results:
        summary_metrics_lstm = {}
        metric_keys = fold_results[0].keys() 
        for key in metric_keys: 
            values = [m[key] for m in fold_results if key in m and m[key] is not None] 
            if not values: 
                summary_metrics_lstm[f'avg_{key}'] = np.nan
                summary_metrics_lstm[f'std_{key}'] = np.nan
                continue
            summary_metrics_lstm[f'avg_{key}'] = np.nanmean(values)
            summary_metrics_lstm[f'std_{key}'] = np.nanstd(values)
        
        for key, value in summary_metrics_lstm.items():
            display_key = key.replace('_', ' ').capitalize()
            if np.isnan(value): print(f"  {display_key}: N/A")
            elif any(metric_name in key for metric_name in ["accuracy", "precision", "recall", "auc", "f1"]): 
                print(f"  {display_key}: {value*100:.2f}% (±{summary_metrics_lstm[key.replace('avg_', 'std_')]*100:.2f}%)")
            else: print(f"  {display_key}: {value:.3f} (±{summary_metrics_lstm[key.replace('avg_', 'std_')]:.3f})") 
        
        lstm_summary_df = pd.DataFrame([summary_metrics_lstm])
        if not os.path.exists(config.RESULTS_PATH):
            os.makedirs(config.RESULTS_PATH)
        lstm_summary_filename = f"lstm_summary_results{'_SMOTE' if USE_SMOTE_LSTM else '_noSMOTE_CW'}.csv"
        lstm_summary_df.to_csv(os.path.join(config.RESULTS_PATH, lstm_summary_filename), index=False)
        print(f"LSTM summary results saved to {os.path.join(config.RESULTS_PATH, lstm_summary_filename)}")
    else:
        print("No LSTM fold results to summarize.")

    print("\n--- End of LSTM Training and Evaluation ---")