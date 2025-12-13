import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from evidently import Report
from evidently.presets import DataDriftPreset
import mlflow
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load and split data
data = pd.read_csv("data/data.csv")
train, test = train_test_split(data, test_size=0.2, stratify=data['species'], random_state=42)

X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = train['species']
X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_test = test['species']

# Load model from MLflow artifacts instead of training
logger.info("Loading model from MLflow artifacts...")
model = mlflow.sklearn.load_model("artifacts/model")
logger.info("Model loaded successfully")

y_pred_original = model.predict(X_test)
original_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_original),
    'precision': precision_score(y_test, y_pred_original, average='weighted'),
    'recall': recall_score(y_test, y_pred_original, average='weighted'),
    'f1': f1_score(y_test, y_pred_original, average='weighted')
}

logger.info(f"Original Test Metrics: {original_metrics}")

# Simulate data drift (shift distributions)
logger.info("\nSimulating data drift...")
X_drifted = X_test.copy()
# Add drift to features (simulate real-world changes)
X_drifted['sepal_length'] = X_drifted['sepal_length'] * 1.2 + 0.5
X_drifted['sepal_width'] = X_drifted['sepal_width'] * 0.8
X_drifted['petal_length'] = X_drifted['petal_length'] * 1.3
# Add noise
X_drifted['petal_width'] = X_drifted['petal_width'] + np.random.normal(0, 0.3, len(X_drifted))

# Evaluate on drifted data
y_pred_drifted = model.predict(X_drifted)
drifted_metrics = {
    'accuracy': accuracy_score(y_test, y_pred_drifted),
    'precision': precision_score(y_test, y_pred_drifted, average='weighted'),
    'recall': recall_score(y_test, y_pred_drifted, average='weighted'),
    'f1': f1_score(y_test, y_pred_drifted, average='weighted')
}

logger.info(f"Drifted Test Metrics: {drifted_metrics}")

# Calculate performance degradation
logger.info("\n" + "="*60)
logger.info("PERFORMANCE IMPACT ANALYSIS")
logger.info("="*60)
for metric in original_metrics:
    original = original_metrics[metric]
    drifted = drifted_metrics[metric]
    degradation = ((original - drifted) / original) * 100
    logger.info(f"{metric.upper()}: {original:.4f} → {drifted:.4f} "
                f"(↓ {degradation:.2f}%)")

# Create comprehensive drift report
logger.info("\nGenerating Evidently drift report...")

# Prepare data for Evidently
reference_data = X_test.copy()
reference_data['target'] = y_test
reference_data['prediction'] = y_pred_original

current_data = X_drifted.copy()
current_data['target'] = y_test
current_data['prediction'] = y_pred_drifted

# Generate comprehensive report
report = Report([DataDriftPreset()], include_tests=True)

# KEY CHANGE: run() returns the evaluation object
my_eval = report.run(reference_data, current_data)

# Save HTML using the returned object
my_eval.save_html("drift_impact_report.html")

logger.info("✓ Comprehensive drift report saved to 'drift_impact_report.html'")

# Get JSON for programmatic access
drift_results = my_eval.dict()

# Summary
logger.info("\n" + "="*60)
logger.info("SUMMARY")
logger.info("="*60)
logger.info(f"Model trained on {len(X_train)} samples")
logger.info(f"Original accuracy: {original_metrics['accuracy']:.4f}")
logger.info(f"Drifted accuracy: {drifted_metrics['accuracy']:.4f}")
logger.info(f"Performance drop: {((original_metrics['accuracy'] - drifted_metrics['accuracy']) / original_metrics['accuracy'] * 100):.2f}%")
logger.info("\nOpen 'drift_impact_report.html' to see detailed drift analysis!")