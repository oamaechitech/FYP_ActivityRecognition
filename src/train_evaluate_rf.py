# src/train_evaluate_rf.py
# Script to train and evaluate a Random Forest model using cross-validation.

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, f1_score
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
        print("Please ensure 'main_processing_pipeline.py' was run for the full dataset.")
        return None, None
    
    try:
        features_df = pd.read_csv(features_path)
        labels_np = np.load(labels_path)
        print("Successfully loaded processed features and labels from:")
        print(f"  Features: {features_path}")
        print(f"  Labels:   {labels_path}")
        print(f"Features shape: {features_df.shape}, Labels shape: {labels_np.shape}")
        return features_df, labels_np
    except Exception as e:
        print(f"Error loading processed data: {e}")
        return None, None

def plot_confusion_matrix(y_true, y_pred, unique_classes_ordered, display_classes, base_title='RF_CV_Confusion_Matrix', cmap=plt.cm.Blues, fold_num=None, smote_status_suffix=""):
    """Plots a confusion matrix and saves it with SMOTE status in filename."""
    cm = confusion_matrix(y_true, y_pred, labels=unique_classes_ordered) 
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=display_classes, yticklabels=display_classes, 
                cbar=True)
    
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
    print(f"Confusion matrix saved to {final_save_path}")
    plt.close()


if __name__ == "__main__":
    USE_SMOTE = False

    smote_file_suffix = "_with_SMOTE" if USE_SMOTE else "_no_SMOTE"

    print(f"--- Random Forest Training and Evaluation with Cross-Validation (SMOTE: {USE_SMOTE}) ---")
    
    X_df, y_full = load_processed_features_and_labels()

    if X_df is None or y_full is None or X_df.empty or y_full.size == 0:
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

    rf_class_weight = None if USE_SMOTE else 'balanced_subsample'

    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=config.RANDOM_SEED,
        class_weight=rf_class_weight, 
        n_jobs=-1 
    )
    
    print(f"\nPerforming Stratified {config.N_FOLDS_CROSS_VALIDATION}-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=config.N_FOLDS_CROSS_VALIDATION, shuffle=True, random_state=config.RANDOM_SEED)
    
    fold_accuracies = []
    fold_f1_macros = []
    fold_f1_fall_class = [] 
    fold_roc_aucs = []
    
    for fold_idx, (train_index, test_index) in enumerate(skf.split(X_full_scaled, y_full)):
        print(f"\n--- Fold {fold_idx + 1}/{config.N_FOLDS_CROSS_VALIDATION} (SMOTE: {USE_SMOTE}) ---")
        
        X_train_fold, X_test_fold = X_full_scaled[train_index], X_full_scaled[test_index]
        y_train_fold, y_test_fold = y_full[train_index], y_full[test_index]
        
        unique_train, counts_train = np.unique(y_train_fold, return_counts=True)
        print(f"Train fold labels before SMOTE: {dict(zip(unique_train, counts_train))}")

        X_train_fold_processed = X_train_fold
        y_train_fold_processed = y_train_fold

        if USE_SMOTE:
            print("Applying SMOTE to the training data of this fold...")
            smote = SMOTE(random_state=config.RANDOM_SEED)
            try:
                minority_class_count = np.min(counts_train)
                smote_k_neighbors = 5
                if minority_class_count <= smote_k_neighbors:
                    smote_k_neighbors = max(1, minority_class_count - 1)
                    print(f"  Adjusted SMOTE k_neighbors to {smote_k_neighbors} due to small minority class size ({minority_class_count}).")
                
                if smote_k_neighbors < 1:
                     print(f"  Skipping SMOTE for fold {fold_idx + 1}: minority class size is too small.")
                else:
                    smote = SMOTE(random_state=config.RANDOM_SEED, k_neighbors=smote_k_neighbors)
                    X_train_fold_processed, y_train_fold_processed = smote.fit_resample(X_train_fold, y_train_fold)
                    unique_smote, counts_smote = np.unique(y_train_fold_processed, return_counts=True)
                    print(f"  Train fold labels after SMOTE: {dict(zip(unique_smote, counts_smote))}")
            except ValueError as e:
                print(f"SMOTE failed for fold {fold_idx + 1}: {e}. Using original training data.")
        
        rf_classifier.fit(X_train_fold_processed, y_train_fold_processed)
        y_pred_fold = rf_classifier.predict(X_test_fold)
        
        acc = accuracy_score(y_test_fold, y_pred_fold)
        f1_macro = f1_score(y_test_fold, y_pred_fold, average='macro', zero_division=0)
        f1_fall = f1_score(y_test_fold, y_pred_fold, pos_label=1, average='binary', zero_division=0) 
        
        fold_accuracies.append(acc)
        fold_f1_macros.append(f1_macro)
        fold_f1_fall_class.append(f1_fall)
        
        print(f"\nClassification Report (Test Set - Fold {fold_idx + 1}, SMOTE: {USE_SMOTE}):")
        print(classification_report(y_test_fold, y_pred_fold, target_names=display_class_names, labels=unique_labels_ordered, digits=3, zero_division=0))
        
        plot_confusion_matrix(y_test_fold, y_pred_fold, 
                              unique_classes_ordered=unique_labels_ordered,
                              display_classes=display_class_names,
                              base_title='RF_CV_Confusion_Matrix', fold_num=fold_idx+1, 
                              smote_status_suffix=smote_file_suffix)

    print("\n--- Cross-Validation Summary ---")
    print(f"SMOTE Used: {USE_SMOTE}") 
    print(f"Average Accuracy: {np.mean(fold_accuracies) * 100:.2f}% (±{np.std(fold_accuracies) * 100:.2f}%)")
    print(f"Average Macro F1-score: {np.mean(fold_f1_macros):.3f} (±{np.std(fold_f1_macros):.3f})")
    print(f"Average Fall Class (1) F1-score: {np.mean(fold_f1_fall_class):.3f} (±{np.std(fold_f1_fall_class):.3f})")
    
    print("\nTraining final RF model on all data for feature importances...")
    final_rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=config.RANDOM_SEED, 
        class_weight=rf_class_weight,
        n_jobs=-1
    )
    
    final_rf_model.fit(X_full_scaled, y_full)

    importances = final_rf_model.feature_importances_
    feature_names = X_df.columns 
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    print("\nTop 20 Feature Importances (from model trained on all data):")
    print(feature_importance_df.head(20))
    
    fi_save_path = os.path.join(config.RESULTS_PATH, f"rf_feature_importances_allData{smote_file_suffix}.csv")
    feature_importance_df.to_csv(fi_save_path, index=False)
    print(f"Feature importances saved to: {fi_save_path}")

    print(f"\n--- End of RF Training and Evaluation (SMOTE: {USE_SMOTE}) ---")