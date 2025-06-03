# src/train_evaluate_models.py (formerly train_evaluate_rf.py)
# Script to train and evaluate multiple machine learning models using cross-validation.

import os
import sys # For sys.exit()
import pandas as pd
import numpy as np
import time # To time model training

from sklearn.model_selection import StratifiedKFold # GridSearchCV removed for this version as not fully implemented previously
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE 

import config
import preprocessing 

def load_processed_features_and_labels() -> tuple[pd.DataFrame | None, np.ndarray | None]:
    """Loads the pre-extracted features and labels for the full dataset."""
    subject_tag_load = "ALL_SUBJECTS" 
    activity_tag_load = "ALL_ACTIVITIES"

    features_filename = f"sisall_features_{subject_tag_load}_{activity_tag_load}_{config.FEATURE_SET_NAME}_w{config.WINDOW_SECONDS}s_o{int(config.WINDOW_OVERLAP_PERCENT*100)}.csv"
    labels_filename = f"sisall_labels_{subject_tag_load}_{activity_tag_load}_{config.FEATURE_SET_NAME}_w{config.WINDOW_SECONDS}s_o{int(config.WINDOW_OVERLAP_PERCENT*100)}.npy"
    
    features_path = os.path.join(config.FEATURES_PATH, features_filename)
    labels_path = os.path.join(config.FEATURES_PATH, labels_filename)

    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print(f"Error: Processed feature/label files not found at {config.FEATURES_PATH}")
        print(f"  Expected features file: {features_filename}")
        print(f"  Expected labels file: {labels_filename}")
        print("Please ensure 'main_processing_pipeline.py' was run and saved files with these names.")
        return None, None
    
    try:
        features_df = pd.read_csv(features_path)
        labels_np = np.load(labels_path)
        print("Successfully loaded processed features and labels.")
        print(f"Features shape: {features_df.shape}, Labels shape: {labels_np.shape}")
        return features_df, labels_np
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None, None

def plot_confusion_matrix(y_true, y_pred, unique_classes_ordered, display_classes, model_name_tag='Model', cmap=plt.cm.Blues, fold_num=None, smote_status_suffix=""):
    """Plots a confusion matrix and saves it with model name and SMOTE status in filename."""
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes_ordered) 
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=display_classes, yticklabels=display_classes, 
                cbar=True)
    
    base_title = f"{model_name_tag}_CV_Confusion_Matrix"
    plot_title = base_title
    save_title = base_title.replace(" ", "_") + smote_status_suffix

    if fold_num is not None:
        plot_title = f"{plot_title} (Fold {fold_num}{smote_status_suffix})"
        save_title = f"{base_title.replace(' ', '_')}_Fold_{fold_num}{smote_status_suffix}"

    plt.title(plot_title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if not os.path.exists(config.RESULTS_PATH):
        os.makedirs(config.RESULTS_PATH)
    
    final_save_path = os.path.join(config.RESULTS_PATH, save_title + ".png")
    plt.savefig(final_save_path)
    print(f"Confusion matrix for {model_name_tag} (Fold {fold_num if fold_num else 'N/A'}) saved to {final_save_path}")
    plt.close()

def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, 
                             unique_classes_ordered, display_class_names, 
                             use_smote_for_this_model, smote_file_suffix, fold_idx_for_plot=None):
    """
    Helper function to train a single model, apply SMOTE if specified, and evaluate.
    Returns a dictionary of metrics.
    """
    metrics = {}
    X_train_processed = X_train.copy() # Use .copy() to avoid modifying original fold data
    y_train_processed = y_train.copy()

    if use_smote_for_this_model:
        print(f"  Applying SMOTE to training data for {model_name}...")
        smote = SMOTE(random_state=config.RANDOM_SEED)
        try:
            minority_class_count = np.min(np.unique(y_train_processed, return_counts=True)[1])
            smote_k_neighbors = 5 
            if minority_class_count <= smote_k_neighbors:
                smote_k_neighbors = max(1, minority_class_count - 1)
            
            if smote_k_neighbors < 1:
                 print(f"    Skipping SMOTE for {model_name}: minority class size ({minority_class_count}) too small.")
            else:
                smote = SMOTE(random_state=config.RANDOM_SEED, k_neighbors=smote_k_neighbors)
                X_train_processed, y_train_processed = smote.fit_resample(X_train_processed, y_train_processed)
        except ValueError as e:
            print(f"    SMOTE failed for {model_name}: {e}. Using original training data.")
    
    start_time = time.time()
    model.fit(X_train_processed, y_train_processed)
    end_time = time.time()
    metrics['training_time'] = end_time - start_time
    
    y_pred = model.predict(X_test)
    
    metrics['accuracy'] = accuracy_score(y_test, y_pred)
    metrics['f1_macro'] = f1_score(y_test, y_pred, average='macro', zero_division=0)
    metrics['f1_fall'] = f1_score(y_test, y_pred, pos_label=1, average='binary', zero_division=0)
    
    report_dict = classification_report(y_test, y_pred, target_names=display_class_names, labels=unique_classes_ordered, output_dict=True, zero_division=0)
    if 'Fall (1)' in report_dict and isinstance(report_dict['Fall (1)'], dict) :
        metrics['recall_fall'] = report_dict['Fall (1)'].get('recall', 0.0)
        metrics['precision_fall'] = report_dict['Fall (1)'].get('precision', 0.0)
    else: 
        metrics['recall_fall'] = 0.0 
        metrics['precision_fall'] = 0.0

    if len(np.unique(y_test)) == 2 and hasattr(model, "predict_proba"):
        try:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
        except Exception as e:
            print(f"    Could not calculate ROC AUC for {model_name}: {e}")
            metrics['roc_auc'] = np.nan
    else:
        metrics['roc_auc'] = np.nan
        
    print(f"\n  Classification Report for {model_name} (SMOTE: {use_smote_for_this_model}):")
    # Use pre-calculated report_dict to avoid recalculating
    print(classification_report(y_test, y_pred, target_names=display_class_names, labels=unique_classes_ordered, digits=3, zero_division=0))
    
    if fold_idx_for_plot is not None: 
        plot_confusion_matrix(y_test, y_pred, 
                              unique_classes_ordered=unique_classes_ordered, 
                              display_classes=display_class_names, 
                              model_name_tag=model_name, fold_num=fold_idx_for_plot, 
                              smote_status_suffix=smote_file_suffix)
    return metrics

if __name__ == "__main__":
    # --- Global Configuration for this script run ---
    USE_SMOTE_GLOBALLY = True # SET THIS TO True or False 
    
    smote_file_suffix = "_with_SMOTE" if USE_SMOTE_GLOBALLY else "_no_SMOTE"
    print(f"--- Model Training and Evaluation with Cross-Validation (Global SMOTE Setting: {USE_SMOTE_GLOBALLY}) ---")
    
    X_df, y_full = load_processed_features_and_labels()

    if X_df is None or y_full is None or X_df.empty or y_full.size == 0 :
        print("Exiting due to data loading issues or empty data.")
        sys.exit(1)

    if y_full.ndim > 1 and y_full.shape[1] == 1:
        y_full = y_full.ravel()
    
    print("\nNormalizing features (MinMax scaling on the full dataset)...")
    X_full_scaled, scaler = preprocessing.normalize_features(X_df.values, method='minmax') 
    if X_full_scaled.size == 0:
        print("Exiting: Feature scaling resulted in empty array.")
        sys.exit(1)
    print("Full dataset feature normalization complete.")

    unique_labels_ordered = np.unique(y_full) 
    display_class_names = ['ADL (0)', 'Fall (1)'] 
    if not (0 in unique_labels_ordered and 1 in unique_labels_ordered and len(unique_labels_ordered) == 2):
        print(f"Warning: Expected binary labels [0, 1] but found {unique_labels_ordered}.")

    models_to_evaluate = {
        "RandomForest": RandomForestClassifier(
            n_estimators=100, 
            random_state=config.RANDOM_SEED, 
            class_weight=None if USE_SMOTE_GLOBALLY else 'balanced_subsample', 
            n_jobs=-1
        ),
        "SVM": SVC(
            random_state=config.RANDOM_SEED, 
            class_weight='balanced' if not USE_SMOTE_GLOBALLY else None, 
            probability=True 
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            max_depth=3,       
            random_state=config.RANDOM_SEED
        ),
        "DecisionTree": DecisionTreeClassifier( # Corrected class_weight logic
            random_state=config.RANDOM_SEED,
            class_weight='balanced' if not USE_SMOTE_GLOBALLY else None 
        ),
        "KNN": KNeighborsClassifier(
            n_neighbors=5 
        )
    }
    
    print(f"\nPerforming Stratified {config.N_FOLDS_CROSS_VALIDATION}-Fold Cross-Validation for all models...")
    skf = StratifiedKFold(n_splits=config.N_FOLDS_CROSS_VALIDATION, shuffle=True, random_state=config.RANDOM_SEED)
    
    all_models_results = {}

    for model_name, model_instance in models_to_evaluate.items():
        print(f"\n===== Evaluating Model: {model_name} (SMOTE: {USE_SMOTE_GLOBALLY}) =====")
        
        fold_metrics_list = [] 
        
        for fold_idx, (train_index, test_index) in enumerate(skf.split(X_full_scaled, y_full)):
            print(f"\n--- {model_name} - Fold {fold_idx + 1}/{config.N_FOLDS_CROSS_VALIDATION} ---")
            
            X_train_fold, X_test_fold = X_full_scaled[train_index], X_full_scaled[test_index]
            y_train_fold, y_test_fold = y_full[train_index], y_full[test_index]
            
            current_fold_metrics = train_and_evaluate_model(
                model_instance, model_name, 
                X_train_fold, y_train_fold, X_test_fold, y_test_fold,
                unique_labels_ordered, display_class_names,
                USE_SMOTE_GLOBALLY, smote_file_suffix,
                fold_idx_for_plot=fold_idx + 1 if fold_idx == 0 else None 
            )
            fold_metrics_list.append(current_fold_metrics)
        
        aggregated_metrics = {}
        if fold_metrics_list: # Ensure list is not empty
            for key in fold_metrics_list[0].keys(): 
                values = [m[key] for m in fold_metrics_list if m is not None and key in m] # Added check for None and key
                if values: # Ensure there are values to process
                    aggregated_metrics[f'avg_{key}'] = np.nanmean(values)
                    aggregated_metrics[f'std_{key}'] = np.nanstd(values)
                else: # Handle case where no valid values were collected for a metric
                    aggregated_metrics[f'avg_{key}'] = np.nan
                    aggregated_metrics[f'std_{key}'] = np.nan
        
        all_models_results[model_name] = aggregated_metrics
        
        print(f"\n--- Summary for {model_name} (SMOTE: {USE_SMOTE_GLOBALLY}) ---")
        for key, value in aggregated_metrics.items():
            display_key = key.replace('_', ' ').capitalize()
            if "time" in key:
                 print(f"  {display_key}: {value:.2f}s")
            elif np.isnan(value):
                 print(f"  {display_key}: N/A")
            elif "accuracy" in key:
                 print(f"  {display_key}: {value*100:.2f}%")
            else:
                 print(f"  {display_key}: {value:.3f}")


    print("\n\n===== OVERALL MODEL COMPARISON SUMMARY =====")
    print(f"SMOTE Used for all models in this run: {USE_SMOTE_GLOBALLY}")
    summary_df = pd.DataFrame.from_dict(all_models_results, orient='index')
    
    # Define columns to format, ensure they exist before formatting
    cols_to_format_percent = ['avg_accuracy']
    cols_to_format_time = ['avg_training_time']
    cols_to_format_float3 = ['avg_f1_macro', 'avg_f1_fall', 'avg_recall_fall', 'avg_precision_fall', 'avg_roc_auc']

    for col in summary_df.columns:
        if col in cols_to_format_percent and summary_df[col].notna().any():
            summary_df[col] = (summary_df[col].astype(float) * 100).round(2).astype(str) + '%'
        elif col in cols_to_format_time and summary_df[col].notna().any():
            summary_df[col] = summary_df[col].astype(float).round(2).astype(str) + 's'
        elif col in cols_to_format_float3 and summary_df[col].notna().any():
             summary_df[col] = summary_df[col].astype(float).round(3)
    
    display_columns = [
        'avg_accuracy', 'avg_f1_macro', 'avg_f1_fall', 
        'avg_recall_fall', 'avg_precision_fall', 'avg_roc_auc', 'avg_training_time'
    ]
    # Ensure all display columns actually exist in summary_df before trying to print
    existing_display_columns = [col for col in display_columns if col in summary_df.columns]
    print(summary_df[existing_display_columns])

    summary_df.to_csv(os.path.join(config.RESULTS_PATH, f"model_comparison_summary{smote_file_suffix}.csv"))
    print(f"Overall model comparison summary saved to {config.RESULTS_PATH}/model_comparison_summary{smote_file_suffix}.csv")

    # --- Feature Importances for Tree-based Models (trained on all data) ---
    tree_based_model_templates = { 
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=config.RANDOM_SEED, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=config.RANDOM_SEED),
        "DecisionTree": DecisionTreeClassifier(random_state=config.RANDOM_SEED) 
    }

    print("\n--- Training Final Tree-Based Models on All Data for Feature Importances ---")
    
    X_train_final_fi = X_full_scaled
    y_train_final_fi = y_full
    if USE_SMOTE_GLOBALLY:
        print("  Applying SMOTE to the full dataset for final feature importance models...")
        smote_fi = SMOTE(random_state=config.RANDOM_SEED)
        try:
            minority_class_count_full = np.min(np.unique(y_full, return_counts=True)[1])
            smote_k_neighbors_full = 5
            if minority_class_count_full <= smote_k_neighbors_full:
                smote_k_neighbors_full = max(1, minority_class_count_full - 1)
            
            if smote_k_neighbors_full < 1:
                print(f"    Skipping SMOTE for final models: minority class size ({minority_class_count_full}) too small.")
            else:
                smote_fi = SMOTE(random_state=config.RANDOM_SEED, k_neighbors=smote_k_neighbors_full)
                X_train_final_fi, y_train_final_fi = smote_fi.fit_resample(X_full_scaled, y_full)
                print(f"    Full dataset shape after SMOTE for FI: {X_train_final_fi.shape}")
        except ValueError as e:
            print(f"    SMOTE failed for full dataset for FI: {e}. Using original full data for FI.")

    for model_name, model_template in tree_based_model_templates.items():
        print(f"\n  Training final {model_name} for feature importances...")
        
        current_model_params = model_template.get_params()
        if model_name == "RandomForest":
            current_model_params['class_weight'] = None if USE_SMOTE_GLOBALLY else 'balanced_subsample'
        elif model_name == "DecisionTree":
            current_model_params['class_weight'] = None if USE_SMOTE_GLOBALLY else 'balanced' # Corrected
        
        final_model = model_template.__class__(**current_model_params) 

        final_model.fit(X_train_final_fi, y_train_final_fi)
        
        if hasattr(final_model, 'feature_importances_'):
            importances = final_model.feature_importances_
            feature_names = X_df.columns 
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

            print(f"\n  Top 20 Feature Importances for {model_name}:")
            print(feature_importance_df.head(20))
            
            fi_savename = f"{model_name.lower()}_feature_importances_allData{smote_file_suffix}.csv"
            fi_save_path = os.path.join(config.RESULTS_PATH, fi_savename)
            feature_importance_df.to_csv(fi_save_path, index=False)
            print(f"  {model_name} feature importances saved to: {fi_save_path}")

    print(f"\n--- End of Model Training and Evaluation (Global SMOTE Setting: {USE_SMOTE_GLOBALLY}) ---")