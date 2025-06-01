# src/train_evaluate_rf.py
# Script to train and evaluate a Random Forest model using cross-validation.

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder # If labels are strings, not needed for 0/1
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE # For SMOTE

# Assuming config.py and preprocessing.py are in the same src/ directory or accessible
import config
import preprocessing # For feature normalization

def load_processed_features_and_labels() -> tuple[pd.DataFrame | None, np.ndarray | None]:
    """Loads the pre-extracted features and labels for the full dataset."""
    # Filenames based on your last output
    features_filename = f"sisall_features_ALL_SUBJECTS_ALL_ACTIVITIES_{config.FEATURE_SET_NAME}_w{config.WINDOW_SECONDS}s_o{int(config.WINDOW_OVERLAP_PERCENT*100)}.csv"
    labels_filename = f"sisall_labels_ALL_SUBJECTS_ALL_ACTIVITIES_{config.FEATURE_SET_NAME}_w{config.WINDOW_SECONDS}s_o{int(config.WINDOW_OVERLAP_PERCENT*100)}.npy"
    
    features_path = os.path.join(config.FEATURES_PATH, features_filename)
    labels_path = os.path.join(config.FEATURES_PATH, labels_filename)

    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print(f"Error: Processed feature/label files not found at {config.FEATURES_PATH}")
        print(f"Expected features: {features_filename}")
        print(f"Expected labels: {labels_filename}")
        print("Please run the main_processing_pipeline.py script for the full dataset first.")
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

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Blues, fold_num=None):
    """Plots a confusion matrix and saves it."""
    cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true)) # ensure labels are passed if y_pred might not contain all classes
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, 
                xticklabels=classes, yticklabels=classes, 
                cbar=True) # Added cbar for clarity
    
    plot_title = title
    save_title = title.replace(" ", "_")
    if fold_num is not None:
        plot_title = f"{title} (Fold {fold_num})"
        save_title = f"{title.replace(' ', '_')}_Fold_{fold_num}"

    plt.title(plot_title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if not os.path.exists(config.RESULTS_PATH):
        os.makedirs(config.RESULTS_PATH)
    
    plt.savefig(os.path.join(config.RESULTS_PATH, save_title + ".png"))
    print(f"Confusion matrix saved to {os.path.join(config.RESULTS_PATH, save_title + '.png')}")
    # plt.show() # Uncomment to display plots interactively during script run
    plt.close() # Close the plot to free up memory, especially in loops

if __name__ == "__main__":
    print("--- Random Forest Training and Evaluation with Cross-Validation ---")
    
    X_df, y_full = load_processed_features_and_labels()

    if X_df is None or y_full is None:
        print("Exiting due to data loading issues.")
        exit()

    # Ensure y is 1D array
    if y_full.ndim > 1 and y_full.shape[1] == 1:
        y_full = y_full.ravel()
    
    # --- Feature Scaling ---
    # Scale features before cross-validation. 
    # It's generally okay to scale the entire dataset before CV for tree-based models like RF,
    # but for some models (like SVM) or techniques that learn parameters (like PCA),
    # scaling should ideally be done *inside* each CV fold (fit on train, transform train & test).
    # For RF, global scaling is usually acceptable and simpler.
    print("\nNormalizing features (MinMax scaling on the full dataset)...")
    # .values to convert DataFrame to NumPy array for scaler
    X_full_scaled, scaler = preprocessing.normalize_features(X_df.values, method='minmax') 
    print("Full dataset feature normalization complete.")

    # --- Model Definition: Random Forest ---
    # Based on your progress report: 100 trees.
    # Add class_weight='balanced' as a good practice for imbalanced data.
    # Grid search for other hyperparameters can be a later step.
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=config.RANDOM_SEED,
        class_weight='balanced', # Helps with imbalanced classes
        n_jobs=-1 # Use all available CPU cores
    )
    
    # --- Stratified K-Fold Cross-Validation ---
    print(f"\nPerforming Stratified {config.N_FOLDS_CROSS_VALIDATION}-Fold Cross-Validation...")
    skf = StratifiedKFold(n_splits=config.N_FOLDS_CROSS_VALIDATION, shuffle=True, random_state=config.RANDOM_SEED)
    
    fold_accuracies = []
    fold_f1_macros = []
    fold_f1_fall_class = [] # F1 score specifically for the "Fall" class (label 1)
    fold_roc_aucs = []
    
    # Define class names based on your labeling (0: ADL, 1: Fall)
    class_names = ['ADL (0)', 'Fall (1)']

    # Option to use SMOTE
    USE_SMOTE = True # Set to False to run without SMOTE

    for fold_idx, (train_index, test_index) in enumerate(skf.split(X_full_scaled, y_full)):
        print(f"\n--- Fold {fold_idx + 1}/{config.N_FOLDS_CROSS_VALIDATION} ---")
        
        X_train_fold, X_test_fold = X_full_scaled[train_index], X_full_scaled[test_index]
        y_train_fold, y_test_fold = y_full[train_index], y_full[test_index]
        
        print(f"Train fold shape: {X_train_fold.shape}, Test fold shape: {X_test_fold.shape}")
        unique_train, counts_train = np.unique(y_train_fold, return_counts=True)
        unique_test, counts_test = np.unique(y_test_fold, return_counts=True)
        print(f"Train fold labels: {dict(zip(unique_train, counts_train))}")
        print(f"Test fold labels: {dict(zip(unique_test, counts_test))}")

        X_train_fold_processed = X_train_fold
        y_train_fold_processed = y_train_fold

        if USE_SMOTE:
            print("Applying SMOTE to the training data of this fold...")
            smote = SMOTE(random_state=config.RANDOM_SEED)
            try:
                X_train_fold_processed, y_train_fold_processed = smote.fit_resample(X_train_fold, y_train_fold)
                print(f"Shape after SMOTE: {X_train_fold_processed.shape}, {y_train_fold_processed.shape}")
                unique_smote, counts_smote = np.unique(y_train_fold_processed, return_counts=True)
                print(f"Train fold labels after SMOTE: {dict(zip(unique_smote, counts_smote))}")
            except ValueError as e:
                print(f"SMOTE failed for fold {fold_idx + 1}: {e}. Using original training data for this fold.")
                # This can happen if a class has too few samples for SMOTE's k_neighbors parameter
        
        # Train the model on the (potentially resampled) training data of this fold
        rf_classifier.fit(X_train_fold_processed, y_train_fold_processed)
        
        # Evaluate on the test data of this fold
        y_pred_fold = rf_classifier.predict(X_test_fold)
        y_pred_proba_fold = rf_classifier.predict_proba(X_test_fold)[:, 1] # Probabilities for positive class (Fall)
        
        acc = accuracy_score(y_test_fold, y_pred_fold)
        f1_macro = f1_score(y_test_fold, y_pred_fold, average='macro')
        f1_fall = f1_score(y_test_fold, y_pred_fold, pos_label=1, average='binary') # F1 for Fall class (label 1)
        
        fold_accuracies.append(acc)
        fold_f1_macros.append(f1_macro)
        fold_f1_fall_class.append(f1_fall)
        
        print(f"Fold {fold_idx + 1} Accuracy: {acc * 100:.2f}%")
        print(f"Fold {fold_idx + 1} Macro F1-score: {f1_macro:.3f}")
        print(f"Fold {fold_idx + 1} Fall Class (1) F1-score: {f1_fall:.3f}")
        
        # ROC AUC only if both classes are present in y_test_fold (common in stratified folds)
        if len(np.unique(y_test_fold)) == 2:
            roc_auc = roc_auc_score(y_test_fold, y_pred_proba_fold)
            fold_roc_aucs.append(roc_auc)
            print(f"Fold {fold_idx + 1} ROC AUC Score: {roc_auc:.3f}")
        else:
            print(f"Fold {fold_idx + 1} ROC AUC Score: N/A (only one class in y_test_fold)")
            # Append NaN or handle appropriately if you average this later
            fold_roc_aucs.append(np.nan) 


        print("\nClassification Report (Test Set - Fold {}):".format(fold_idx + 1))
        print(classification_report(y_test_fold, y_pred_fold, target_names=class_names, digits=3, zero_division=0))
        
        # Plot confusion matrix for the first fold as an example, or for all if desired
        if fold_idx == 0: # Or remove this condition to plot for every fold
             plot_confusion_matrix(y_test_fold, y_pred_fold, classes=class_names, 
                                   title='RF_CV_Confusion_Matrix', fold_num=fold_idx+1)

    # --- Averaged Results from Cross-Validation ---
    print("\n--- Cross-Validation Summary ---")
    print(f"Average Accuracy: {np.mean(fold_accuracies) * 100:.2f}% (+/- {np.std(fold_accuracies) * 100:.2f}%)")
    print(f"Average Macro F1-score: {np.mean(fold_f1_macros):.3f} (+/- {np.std(fold_f1_macros):.3f})")
    print(f"Average Fall Class (1) F1-score: {np.mean(fold_f1_fall_class):.3f} (+/- {np.std(fold_f1_fall_class):.3f})")
    
    # Average ROC AUC, ignoring NaNs if any fold had only one class in y_test
    avg_roc_auc = np.nanmean(fold_roc_aucs)
    std_roc_auc = np.nanstd(fold_roc_aucs)
    if not np.isnan(avg_roc_auc):
        print(f"Average ROC AUC Score: {avg_roc_auc:.3f} (+/- {std_roc_auc:.3f})")
    else:
        print("Average ROC AUC Score: N/A (could not be computed for all folds)")


    # Optional: Train final model on ALL scaled data (X_full_scaled, y_full) for feature importances
    # This model is not for generalization estimate but for inspection.
    print("\nTraining final RF model on all data for feature importances...")
    final_rf_model = RandomForestClassifier(
        n_estimators=100, 
        random_state=config.RANDOM_SEED, 
        class_weight='balanced', 
        n_jobs=-1
    ).fit(X_full_scaled, y_full)

    importances = final_rf_model.feature_importances_
    feature_names = X_df.columns # Get feature names from the original DataFrame
    feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)

    print("\nTop 20 Feature Importances:")
    print(feature_importance_df.head(20))
    
    # Save feature importances
    fi_save_path = os.path.join(config.RESULTS_PATH, "rf_feature_importances.csv")
    feature_importance_df.to_csv(fi_save_path, index=False)
    print(f"Feature importances saved to: {fi_save_path}")

    print("\n--- End of RF Training and Evaluation with Cross-Validation ---")