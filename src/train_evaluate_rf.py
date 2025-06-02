# src/train_evaluate_rf.py
# Script to train and evaluate a Random Forest model using cross-validation.

import os
import sys # For sys.exit()
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier0
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE # For SMOTE

# Assuming config.py and preprocessing.py are in the same src/ directory or accessible
import config
import preprocessing # For feature normalization

def load_processed_features_and_labels() -> tuple[pd.DataFrame | None, np.ndarray | None]:
    """Loads the pre-extracted features and labels for the full dataset."""
    # Filenames should match how they are saved by main_processing_pipeline.py
    # This assumes you processed ALL subjects and ALL activities
    subject_tag_load = "ALL_SUBJECTS" 
    activity_tag_load = "ALL_ACTIVITIES"

    # Construct filenames based on how main_processing_pipeline.py saves them
    features_filename = f"sisall_features_{subject_tag_load}_{activity_tag_load}_{config.FEATURE_SET_NAME}_w{config.WINDOW_SECONDS}s_o{int(config.WINDOW_OVERLAP_PERCENT*100)}.csv"
    labels_filename = f"sisall_labels_{subject_tag_load}_{activity_tag_load}_{config.FEATURE_SET_NAME}_w{config.WINDOW_SECONDS}s_o{int(config.WINDOW_OVERLAP_PERCENT*100)}.npy"
    
    features_path = os.path.join(config.FEATURES_PATH, features_filename)
    labels_path = os.path.join(config.FEATURES_PATH, labels_filename)

    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print(f"Error: Processed feature/label files not found at {config.FEATURES_PATH}")
        print(f"  Expected features file: {features_filename}")
        print(f"  Expected labels file: {labels_filename}")
        print("Please ensure 'main_processing_pipeline.py' was run for the full dataset and saved files with these names.")
        print(f"Also check that 'config.FEATURE_SET_NAME' is currently: '{config.FEATURE_SET_NAME}'")
        return None, None
    
    try:
        features_df = pd.read_csv(features_path)
        labels_np = np.load(labels_path)
        print(f"Successfully loaded processed features and labels from:")
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
    # Filename construction: use replace only on base_title to avoid issues if suffix has spaces
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
    plt.close() # Close the plot to free up memory, especially in loops

if __name__ == "__main__":
    # --- Configuration for this specific run ---
    # CHANGE THIS FLAG TO RUN WITH OR WITHOUT SMOTE
    USE_SMOTE = False # Set to True to run with SMOTE, False to run without SMOTE

    # Determine Suffix for output filenames based on SMOTE status
    smote_file_suffix = "_with_SMOTE" if USE_SMOTE else "_no_SMOTE"
    # --- END OF FILENAME SUFFIX DETERMINATION ---

    print(f"--- Random Forest Training and Evaluation with Cross-Validation (SMOTE: {USE_SMOTE}) ---")
    
    X_df, y_full = load_processed_features_and_labels()

    if X_df is None or y_full is None or X_df.empty or y_full.size == 0 :
        print("Exiting due to data loading issues or empty data.")
        sys.exit(1) # Exit with a non-zero code to indicate an error

    if y_full.ndim > 1 and y_full.shape[1] == 1:
        y_full = y_full.ravel()
    
    print("\nNormalizing features (MinMax scaling on the full dataset)...")
    X_full_scaled, scaler = preprocessing.normalize_features(X_df.values, method='minmax') 
    if X_full_scaled.size == 0:
        print("Exiting: Feature scaling resulted in empty array.")
        sys.exit(1)
    print("Full dataset feature normalization complete.")

    # Define class names based on your labeling (0: ADL, 1: Fall)
    # and the unique labels present in y_full for confusion matrix plotting
    unique_labels_ordered = np.unique(y_full) # e.g., array([0, 1])
    display_class_names = ['ADL (0)', 'Fall (1)'] # For display purposes
    if not (0 in unique_labels_ordered and 1 in unique_labels_ordered and len(unique_labels_ordered) == 2):
        print(f"Warning: Expected binary labels [0, 1] but found {unique_labels_ordered}. Confusion matrix and some metrics might be affected if not binary.")
        # Adjust display_class_names if necessary, or handle as multi-class if that's the case.
        # For now, proceeding with assumption of 0 and 1.


    # Determine class_weight argument for RandomForestClassifier
    # If using SMOTE, it's often recommended to NOT use class_weight='balanced' simultaneously,
    # as SMOTE already handles the imbalance. If not using SMOTE, 'balanced' or 'balanced_subsample' is good.
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
        
        print(f"Train fold shape: {X_train_fold.shape}, Test fold shape: {X_test_fold.shape}")
        unique_train, counts_train = np.unique(y_train_fold, return_counts=True)
        unique_test, counts_test = np.unique(y_test_fold, return_counts=True)
        print(f"Train fold labels before SMOTE: {dict(zip(unique_train, counts_train))}")
        print(f"Test fold labels: {dict(zip(unique_test, counts_test))}")

        X_train_fold_processed = X_train_fold
        y_train_fold_processed = y_train_fold

        if USE_SMOTE:
            print("Applying SMOTE to the training data of this fold...")
            smote = SMOTE(random_state=config.RANDOM_SEED)
            try:
                # SMOTE's k_neighbors must be less than the number of samples in the smallest class
                # Check minority class size
                minority_class_count = np.min(counts_train)
                smote_k_neighbors = 5 # Default for SMOTE
                if minority_class_count <= smote_k_neighbors: # k_neighbors must be < n_samples in minority class
                    smote_k_neighbors = max(1, minority_class_count - 1) # Adjust k_neighbors
                    print(f"  Adjusted SMOTE k_neighbors to {smote_k_neighbors} due to small minority class size ({minority_class_count}).")
                
                if smote_k_neighbors < 1 : # Cannot run SMOTE if k_neighbors is less than 1
                     print(f"  Skipping SMOTE for fold {fold_idx + 1}: minority class size ({minority_class_count}) too small for k_neighbors adjustment.")
                else:
                    smote = SMOTE(random_state=config.RANDOM_SEED, k_neighbors=smote_k_neighbors)
                    X_train_fold_processed, y_train_fold_processed = smote.fit_resample(X_train_fold, y_train_fold)
                    print(f"  Shape after SMOTE: {X_train_fold_processed.shape}, {y_train_fold_processed.shape}")
                    unique_smote, counts_smote = np.unique(y_train_fold_processed, return_counts=True)
                    print(f"  Train fold labels after SMOTE: {dict(zip(unique_smote, counts_smote))}")

            except ValueError as e:
                print(f"SMOTE failed for fold {fold_idx + 1}: {e}. Using original training data for this fold.")
        
        rf_classifier.fit(X_train_fold_processed, y_train_fold_processed)
        
        y_pred_fold = rf_classifier.predict(X_test_fold)
        
        acc = accuracy_score(y_test_fold, y_pred_fold)
        f1_macro = f1_score(y_test_fold, y_pred_fold, average='macro', zero_division=0)
        # Ensure pos_label=1 for Fall class if your labels are 0 (ADL) and 1 (Fall)
        f1_fall = f1_score(y_test_fold, y_pred_fold, pos_label=1, average='binary', zero_division=0) 
        
        fold_accuracies.append(acc)
        fold_f1_macros.append(f1_macro)
        fold_f1_fall_class.append(f1_fall)
        
        print(f"Fold {fold_idx + 1} Accuracy: {acc * 100:.2f}%")
        print(f"Fold {fold_idx + 1} Macro F1-score: {f1_macro:.3f}")
        print(f"Fold {fold_idx + 1} Fall Class (1) F1-score: {f1_fall:.3f}")
        
        # ROC AUC
        if len(np.unique(y_test_fold)) == 2: # Only if binary and both classes present in y_test_fold
            y_pred_proba_fold = rf_classifier.predict_proba(X_test_fold)[:, 1]
            roc_auc = roc_auc_score(y_test_fold, y_pred_proba_fold)
            fold_roc_aucs.append(roc_auc)
            print(f"Fold {fold_idx + 1} ROC AUC Score: {roc_auc:.3f}")
        else:
            print(f"Fold {fold_idx + 1} ROC AUC Score: N/A (test data for fold not binary or only one class present)")
            fold_roc_aucs.append(np.nan) 

        print(f"\nClassification Report (Test Set - Fold {fold_idx + 1}, SMOTE: {USE_SMOTE}):")
        print(classification_report(y_test_fold, y_pred_fold, target_names=display_class_names, labels=unique_labels_ordered, digits=3, zero_division=0))
        
        # Plot confusion matrix for this fold
        plot_confusion_matrix(y_test_fold, y_pred_fold, 
                              unique_classes_ordered=unique_labels_ordered, # Pass the actual unique label values
                              display_classes=display_class_names, # Pass the display names
                              base_title='RF_CV_Confusion_Matrix', fold_num=fold_idx+1, 
                              smote_status_suffix=smote_file_suffix)

    print("\n--- Cross-Validation Summary ---")
    print(f"SMOTE Used: {USE_SMOTE}") 
    print(f"Average Accuracy: {np.mean(fold_accuracies) * 100:.2f}% (+/- {np.std(fold_accuracies) * 100:.2f}%)")
    print(f"Average Macro F1-score: {np.mean(fold_f1_macros):.3f} (+/- {np.std(fold_f1_macros):.3f})")
    print(f"Average Fall Class (1) F1-score: {np.mean(fold_f1_fall_class):.3f} (+/- {np.std(fold_f1_fall_class):.3f})")
    
    avg_roc_auc = np.nanmean(fold_roc_aucs)
    std_roc_auc = np.nanstd(fold_roc_aucs)
    if not np.isnan(avg_roc_auc):
        print(f"Average ROC AUC Score: {avg_roc_auc:.3f} (+/- {std_roc_auc:.3f})")
    else:
        print("Average ROC AUC Score: N/A (could not be computed reliably across all folds)")

    print("\nTraining final RF model on all data for feature importances...")
    final_rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=config.RANDOM_SEED, 
        class_weight=rf_class_weight, # Use the same class_weight logic as in CV
        n_jobs=-1
    )
    
    # For feature importances, it's generally better to train on the original (but scaled)
    # full dataset to get a sense of feature importance on the true data distribution.
    # If SMOTE was used in CV, the feature importances might be skewed by synthetic samples.
    # Alternatively, average feature importances from each fold's model if SMOTE was used.
    # Here, we train on the full scaled original dataset:
    final_rf_model.fit(X_full_scaled, y_full)

    importances = final_rf_model.feature_importances_
    feature_names = X_df.columns 
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    print("\nTop 20 Feature Importances (from model trained on all original scaled data):")
    print(feature_importance_df.head(20))
    
    fi_save_path = os.path.join(config.RESULTS_PATH, f"rf_feature_importances_allData{smote_file_suffix}.csv")
    feature_importance_df.to_csv(fi_save_path, index=False)
    print(f"Feature importances (all data model) saved to: {fi_save_path}")

    print(f"\n--- End of RF Training and Evaluation with Cross-Validation (SMOTE: {USE_SMOTE}) ---")