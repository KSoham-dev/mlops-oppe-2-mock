"""
Label Poisoning Attack Simulation

This script trains models with different levels of label corruption to analyze
the impact of data poisoning attacks on model performance.
Uses the same structure and parameters as train.py for consistency.
"""

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import mlflow
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MLflow configuration (same as train.py)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
experiment_name = "iris-classifier-poisoning-analysis"
artifact_location = "gs://mlops-course-clean-vista-473214-i6/mlflow-assets/iris-classifier-model"
registered_model_name = "iris-classifier-poisoned"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Create/set experiment
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    logger.info(f"Creating new experiment '{experiment_name}'...")
    mlflow.create_experiment(experiment_name, artifact_location=artifact_location)
    mlflow.set_experiment(experiment_name)
else:
    logger.info(f"Using existing experiment '{experiment_name}'")
    mlflow.set_experiment(experiment_name)

# Load data (same as train.py)
data = pd.read_csv("data/data.csv")

# Split data (same as train.py)
train, test = train_test_split(data, test_size=0.2, stratify=data['species'], random_state=42)
X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = train.species
X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_test = test.species

# Same hyperparameter grid as train.py
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 2, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6],
    'class_weight': [None, 'balanced']
}

# Label corruption levels to test
CORRUPTION_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30]


def corrupt_labels(y_train_orig, corruption_percentage):
    """
    Corrupt a percentage of training labels by changing them to other classes
    
    Args:
        y_train_orig: Original training labels (pandas Series)
        corruption_percentage: Percentage of labels to corrupt (0.0 to 1.0)
    
    Returns:
        Corrupted labels as pandas Series
    """
    y_corrupted = y_train_orig.copy()
    n_corrupt = int(len(y_train_orig) * corruption_percentage)
    
    if n_corrupt > 0:
        # Randomly select indices to corrupt
        corrupt_indices = np.random.choice(y_train_orig.index, n_corrupt, replace=False)
        
        species_list = y_train_orig.unique()
        
        for idx in corrupt_indices:
            current_label = y_corrupted.loc[idx]
            # Change to a different random label
            other_labels = [s for s in species_list if s != current_label]
            y_corrupted.loc[idx] = np.random.choice(other_labels)
    
    return y_corrupted


def train_with_corruption(corruption_level):
    """
    Train model with specified level of label corruption
    
    Args:
        corruption_level: Percentage of labels to corrupt (0.0 to 1.0)
    """
    run_name = f"Clean_Data" if corruption_level == 0.0 else f"Poisoned_{int(corruption_level*100)}pct"
    
    logger.info(f"\n{'='*80}")
    logger.info(f"Training with {corruption_level*100:.0f}% label corruption")
    logger.info(f"{'='*80}")
    
    # Enable autologging (same as train.py)
    mlflow.sklearn.autolog(
        max_tuning_runs=10,
        registered_model_name=None  # Don't register poisoned models
    )
    
    with mlflow.start_run(run_name=run_name):
        # Corrupt labels
        y_train_corrupted = corrupt_labels(y_train, corruption_level)
        
        # Count actual corruptions
        n_corrupted = (y_train != y_train_corrupted).sum()
        logger.info(f"Corrupted {n_corrupted} labels out of {len(y_train)}")
        
        # Log corruption details
        mlflow.log_param("corruption_level", corruption_level)
        mlflow.log_metric("labels_corrupted", n_corrupted)
        
        # Train model (same approach as train.py)
        logger.info("Initializing GridSearchCV...")
        clf = DecisionTreeClassifier(random_state=42)
        grid_search = GridSearchCV(
            clf, param_grid, cv=5, scoring="accuracy", n_jobs=-1, verbose=1
        )
        
        logger.info("Fitting model...")
        grid_search.fit(X_train, y_train_corrupted)
        
        # Evaluate on clean test set
        test_score = grid_search.score(X_test, y_test)
        
        # Log metrics explicitly (same as train.py)
        mlflow.log_metric("test_accuracy", test_score)
        mlflow.log_metric("cv_accuracy", grid_search.best_score_)
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best cross-validation score: {grid_search.best_score_:.3f}")
        logger.info(f"Test score: {test_score:.3f}")
        
        logger.info(f"Training completed successfully")


def main():
    """Main execution function"""
    logger.info("\n" + "="*80)
    logger.info("LABEL POISONING ATTACK SIMULATION")
    logger.info("="*80 + "\n")
    
    # Train models with different corruption levels
    for corruption_level in CORRUPTION_LEVELS:
        train_with_corruption(corruption_level)
    
    logger.info("\n" + "="*80)
    logger.info("ANALYSIS COMPLETE - Check MLflow UI for results")
    logger.info("="*80 + "\n")


if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)