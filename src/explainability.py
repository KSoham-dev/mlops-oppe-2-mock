import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

# Load model and data
model = joblib.load("artifacts/model/model.pkl")
if hasattr(model, "best_estimator_"):
    model = model.best_estimator_

data = pd.read_csv("data/data.csv")
X = data[["sepal_length", "sepal_width", "petal_length", "petal_width"]]

# Create explainer and calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

# Create output directory
os.makedirs("shap_plots", exist_ok=True)

print(f"SHAP values shape: {shap_values.shape}")

# Number of classes
num_classes = 3
class_names = ["setosa", "versicolor", "virginica"]

# Generate beeswarm plots for each class
for i in range(num_classes):
    plt.figure()
    shap.summary_plot(shap_values[:, :, i], X, show=False)
    plt.title(f"SHAP Summary - {class_names[i]}", fontweight="bold")
    plt.savefig(
        f"shap_plots/shap_summary_beeswarm_{class_names[i]}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
    print(f"  ✓ Saved: shap_summary_beeswarm_{class_names[i]}.png")

# Generate bar plots for each class
for i in range(num_classes):
    plt.figure()
    shap.summary_plot(shap_values[:, :, i], X, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance - {class_names[i]}", fontweight="bold")
    plt.savefig(
        f"shap_plots/shap_summary_bar_{class_names[i]}.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()
    print(f"  ✓ Saved: shap_summary_bar_{class_names[i]}.png")

print("\nAll SHAP plots saved to shap_plots/")
print("  - Beeswarm plots: Show feature effects for each class")
print("  - Bar plots: Show feature importance for each class")

# Generate markdown report for CML
with open("shap_report.md", "w") as f:
    f.write("## 🔍 SHAP Explainability Analysis\n\n")
    f.write(
        "SHAP (SHapley Additive exPlanations) values explain the "
        "contribution of each feature to the model's predictions.\n\n"
    )

    for i, class_name in enumerate(class_names):
        f.write(f"### {class_name.capitalize()} Class\n\n")
        f.write("#### Feature Impact (Beeswarm Plot)\n")
        f.write(
            f"![SHAP Beeswarm - {class_name}]"
            f"(shap_plots/shap_summary_beeswarm_{class_name}.png)\n\n"
        )
        f.write("#### Feature Importance (Bar Plot)\n")
        f.write(
            f"![SHAP Bar - {class_name}]"
            f"(shap_plots/shap_summary_bar_{class_name}.png)\n\n"
        )

print("\n✓ SHAP report saved to shap_report.md")
