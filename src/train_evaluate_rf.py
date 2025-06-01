# src/train_evaluate_rf.py
# Script to train and evaluate a Random Forest model.

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder # If labels are strings
import matplotlib.pyplot as plt
import seaborn as sns

import config
import preprocessing # For feature normalization if needed

def load_processed_features_and_labels() -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    """Loads the pre-extracted features and labels."""
    features_filename = f"sisall_features_{config.FEATURE_SET_NAME}_w{config.WINDOW_SECONDS}s_o{int(config.WINDOW_OVERLAP_PERCENT*100)}.csv"
    labels_filename = f"sisall_labels_{config.FEATURE_SET_NAME}_w{config.WINDOW_SECONDS}s_o{int(config.WINDOW_OVERLAP_PERCENT*100)}.npy"
    
    features_path = os.path.join(config.FEATURES_PATH, features_filename)
    labels_path = os.path.join(config.FEATURES_PATH, labels_filename)

    if not os.path.exists(features_path) or not os.path.exists(labels_path):
        print(f"Error: Processed feature/label files not found at {config.FEATURES_PATH}")
        print(f"Expected features: {features_filename}")
        print(f"Expected labels: {labels_filename}")
        print("Please run the main_processing_pipeline.py script first.")
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

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', cmap=plt.cm.Blues):
    """Plots a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # Ensure results directory exists
    if not os.path.exists(config.RESULTS_PATH):
        os.makedirs(config.RESULTS_PATH)
    plt.savefig(os.path.join(config.RESULTS_PATH, title.replace(" ", "_") + ".png"))
    print(f"Confusion matrix saved to {config.RESULTS_PATH}")
    # plt.show() # Uncomment if you want to display plots interactively

if __name__ == "__main__":
    print("--- Random Forest Training and Evaluation ---")
    
    X, y = load_processed_features_and_labels()

    if X is None or y is None:
        print("Exiting due to data loading issues.")
        exit()

    # Ensure y is 1D array
    if y.ndim > 1 and y.shape[1] == 1:
        y = y.ravel()
    
    # --- Data Splitting (Consistent with your Progress Report initial split) ---
    # Later, replace this with K-Fold Cross-Validation for robust evaluation
    # For now, a simple train-test split to replicate progress report baseline
    print(f"\nSplitting data into train/test (Test ratio: {config.TRAIN_TEST_SPLIT_RATIO})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TRAIN_TEST_SPLIT_RATIO, 
        random_state=config.RANDOM_SEED,
        stratify=y # Important for imbalanced datasets
    )
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # --- Feature Normalization/Scaling (Optional but often recommended for RF too) ---
    # Zurbuchen et al. (2021) normalized features before ML.
    # If your features have very different scales, this can be beneficial.
    print("\nNormalizing features (MinMax scaling)...")
    X_train_scaled, X_test_scaled, scaler = preprocessing.normalize_features(
        X_train.values, X_test.values, method='minmax' # .values to convert DataFrame to NumPy array
    )
    # Convert scaled numpy arrays back to DataFrames if needed by RF, or use numpy arrays directly
    # X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    # X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    print("Feature normalization complete.")


    # --- Model Training: Random Forest ---
    # Based on your progress report: 100 trees, hyperparameters tuned using grid search.
    # For now, let's use a basic RF. Grid search will be a later step.
    print("\nTraining Random Forest Classifier...")
    rf_classifier = RandomForestClassifier(
        n_estimators=100,       # Consistent with your progress report
        random_state=config.RANDOM_SEED,
        # Add other parameters if you recall them from your grid search, e.g.:
        # max_depth=None, 
        # min_samples_split=2,
        # min_samples_leaf=1,
        class_weight='balanced' # Good for imbalanced data; or use SMOTE on training data
    )
    
    # Train on scaled data
    rf_classifier.fit(X_train_scaled, y_train)
    print("Random Forest training complete.")

    # --- Model Evaluation on Test Set ---
    print("\nEvaluating model on the test set...")
    y_pred_test = rf_classifier.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred_test)
    # Your progress report had 85% accuracy, 82% precision, 84% recall.
    # These values here will depend on your full pipeline.
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    # Detailed classification report
    # Assuming labels are 0 (ADL) and 1 (Fall) as per `process_dataset`
    target_names = ['ADL (0)', 'Fall (1)'] 
    report = classification_report(y_test, y_pred_test, target_names=target_names, digits=3)
    print("\nClassification Report (Test Set):")
    print(report)

    # Confusion Matrix
    plot_confusion_matrix(y_test, y_pred_test, classes=target_names, title='RF Confusion Matrix (Test Set)')

    # ROC AUC Score (if binary classification)
    if len(np.unique(y)) == 2: # Check if it's binary classification
        y_pred_proba_test = rf_classifier.predict_proba(X_test_scaled)[:, 1] # Probabilities for the positive class
        roc_auc = roc_auc_score(y_test, y_pred_proba_test)
        print(f"Test ROC AUC Score: {roc_auc:.3f}")

    # --- TODO for Final Project: K-Fold Cross-Validation ---
    # For more robust evaluation, implement K-Fold (e.g., StratifiedKFold or GroupKFold for subject independence)
    # Example (conceptual, would need subject groups for GroupKFold):
    # print("\nPerforming Stratified K-Fold Cross-Validation...")
    # skf = StratifiedKFold(n_splits=config.N_FOLDS_CROSS_VALIDATION, shuffle=True, random_state=config.RANDOM_SEED)
    # # Use the full scaled dataset (X_scaled, y) for cross-validation
    # X_full_scaled, _ = preprocessing.normalize_features(X.values, method='minmax')
    # cv_scores_f1 = cross_val_score(rf_classifier, X_full_scaled, y, cv=skf, scoring='f1_macro') # or 'f1_weighted' or 'f1' for binary positive class
    # print(f"Macro F1-scores for {config.N_FOLDS_CROSS_VALIDATION} folds: {cv_scores_f1}")
    # print(f"Average Macro F1-score (CV): {np.mean(cv_scores_f1) * 100:.2f}% (+/- {np.std(cv_scores_f1) * 2 * 100:.2f}%)")

    print("\n--- End of RF Training and Evaluation ---")