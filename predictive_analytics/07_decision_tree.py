from pathlib import Path
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree,
    export_text
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# ==================================================
# STEP 7: DECISION TREE CLASSIFICATION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
PLOT_DIR = BASE_DIR / "plots"

OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)

df = pd.read_csv(
    OUTPUT_DIR / "t20wc_classification_ready.csv"
)

print("\n========== DATASET INFORMATION ==========")
print("Dataset Shape:", df.shape)
print("Missing Values:", df.isnull().sum().sum())

# --------------------------------------------------
# 1. SELECT FEATURES AND TARGET
# --------------------------------------------------

feature_columns = ["Season", "Innings", "Team"]
target_column = "Performance"

X = df[feature_columns].copy()
y = df[target_column].copy()

# Treat season and innings as categories.
X["Season"] = X["Season"].astype(str)
X["Innings"] = X["Innings"].astype(str)
X["Team"] = X["Team"].astype(str)

labels = ["Low", "Medium", "High"]

print("\n========== TARGET DISTRIBUTION ==========")
print(y.value_counts().reindex(labels, fill_value=0))

# --------------------------------------------------
# 2. TRAIN-TEST SPLIT
# --------------------------------------------------
# Stratification preserves class proportions.

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n========== TRAIN-TEST SPLIT ==========")
print("Training Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# --------------------------------------------------
# 3. ONE-HOT ENCODING
# --------------------------------------------------

encoder = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        ("categorical", encoder, feature_columns)
    ]
)

# --------------------------------------------------
# 4. DECISION TREE MODEL
# --------------------------------------------------

classifier = DecisionTreeClassifier(
    criterion="gini",
    max_depth=5,
    min_samples_leaf=10,
    class_weight="balanced",
    random_state=42
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", classifier)
])

start_time = time.perf_counter()

model.fit(X_train, y_train)

training_time = time.perf_counter() - start_time

# --------------------------------------------------
# 5. PREDICTION
# --------------------------------------------------

start_time = time.perf_counter()

y_pred = model.predict(X_test)

prediction_time = time.perf_counter() - start_time

# --------------------------------------------------
# 6. PERFORMANCE EVALUATION
# --------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

balanced_accuracy = balanced_accuracy_score(
    y_test, y_pred
)

report = classification_report(
    y_test,
    y_pred,
    labels=labels,
    zero_division=0,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

report_df.to_csv(
    OUTPUT_DIR / "decision_tree_classification_report.csv"
)

print("\n========== MODEL PERFORMANCE ==========")
print("Accuracy:", round(accuracy, 4))
print("Balanced Accuracy:", round(balanced_accuracy, 4))
print("Training Time:", round(training_time, 6))
print("Prediction Time:", round(prediction_time, 6))

print("\n========== CLASSIFICATION REPORT ==========")
print(classification_report(
    y_test,
    y_pred,
    labels=labels,
    zero_division=0
))

# --------------------------------------------------
# 7. CONFUSION MATRIX
# --------------------------------------------------

cm = confusion_matrix(
    y_test, y_pred, labels=labels
)

print("\n========== CONFUSION MATRIX ==========")
print(cm)

fig, ax = plt.subplots(figsize=(8, 6))

ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=labels
).plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=False
)

ax.set_title("Decision Tree - Confusion Matrix")
fig.tight_layout()

fig.savefig(
    PLOT_DIR / "13_decision_tree_confusion_matrix.png",
    dpi=300
)
fig.savefig(
    PLOT_DIR / "13_decision_tree_confusion_matrix.pdf"
)

plt.close(fig)

# --------------------------------------------------
# 8. FEATURE IMPORTANCE
# --------------------------------------------------

feature_names = (
    model.named_steps["preprocessor"]
    .get_feature_names_out()
)

tree_model = model.named_steps["classifier"]

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": tree_model.feature_importances_
})

importance_df = importance_df.sort_values(
    "Importance", ascending=False
)

importance_df.to_csv(
    OUTPUT_DIR / "decision_tree_feature_importance.csv",
    index=False
)

print("\n========== FEATURE IMPORTANCE ==========")
print(importance_df.head(15).to_string(index=False))

top_features = importance_df.head(15).sort_values(
    "Importance"
)

fig, ax = plt.subplots(figsize=(11, 7))

sns.barplot(
    data=top_features,
    x="Importance",
    y="Feature",
    color="steelblue",
    ax=ax
)

ax.set_title("Decision Tree - Top 15 Feature Importances")
ax.set_xlabel("Gini Feature Importance")
ax.set_ylabel("Encoded Features")

fig.tight_layout()

fig.savefig(
    PLOT_DIR / "14_decision_tree_feature_importance.png",
    dpi=300
)
fig.savefig(
    PLOT_DIR / "14_decision_tree_feature_importance.pdf"
)

plt.close(fig)

# --------------------------------------------------
# 9. VISUALIZE DECISION TREE
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(24, 12))

plot_tree(
    tree_model,
    feature_names=feature_names,
    class_names=tree_model.classes_,
    filled=True,
    rounded=True,
    fontsize=8,
    max_depth=3,
    ax=ax
)

ax.set_title(
    "Decision Tree Classification (First 4 Levels)",
    fontsize=18
)

fig.tight_layout()

fig.savefig(
    PLOT_DIR / "15_decision_tree_structure.png",
    dpi=300,
    bbox_inches="tight"
)
fig.savefig(
    PLOT_DIR / "15_decision_tree_structure.pdf",
    bbox_inches="tight"
)

plt.close(fig)

# --------------------------------------------------
# 10. GENERATE IF-THEN RULES
# --------------------------------------------------

rules_text = export_text(
    tree_model,
    feature_names=list(feature_names),
    max_depth=5,
    decimals=3
)

print("\n========== DECISION TREE IF-THEN RULES ==========")
print(rules_text)

with open(
    OUTPUT_DIR / "decision_tree_rules.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(rules_text)

# --------------------------------------------------
# 11. SAVE ACTUAL VS PREDICTED
# --------------------------------------------------

predictions_df = X_test.copy()

predictions_df["Actual"] = y_test.values
predictions_df["Predicted"] = y_pred

predictions_df["Correct"] = (
    predictions_df["Actual"]
    == predictions_df["Predicted"]
)

predictions_df.to_csv(
    OUTPUT_DIR / "decision_tree_predictions.csv",
    index=False
)

# --------------------------------------------------
# 12. MAJORITY CLASS BASELINE
# --------------------------------------------------

majority_class = y_train.value_counts().idxmax()

baseline_predictions = np.full(
    len(y_test), majority_class
)

baseline_accuracy = accuracy_score(
    y_test, baseline_predictions
)

baseline_balanced_accuracy = balanced_accuracy_score(
    y_test, baseline_predictions
)

print("\n========== BASELINE COMPARISON ==========")
print("Majority Class:", majority_class)
print("Baseline Accuracy:", round(baseline_accuracy, 4))
print("Baseline Balanced Accuracy:",
      round(baseline_balanced_accuracy, 4))
print("Decision Tree Accuracy:", round(accuracy, 4))
print("Decision Tree Balanced Accuracy:",
      round(balanced_accuracy, 4))

# --------------------------------------------------
# 13. SAVE SUMMARY
# --------------------------------------------------

summary = f"""
DECISION TREE CLASSIFICATION SUMMARY
=====================================

Total Records: {len(df)}
Training Records: {len(X_train)}
Testing Records: {len(X_test)}

Features: {feature_columns}
Target: {target_column}

Criterion: Gini
Maximum Depth: 5
Minimum Samples per Leaf: 10
Class Weight: Balanced
Random State: 42

Accuracy: {accuracy:.4f}
Balanced Accuracy: {balanced_accuracy:.4f}

Macro F1 Score: {report['macro avg']['f1-score']:.4f}
Weighted F1 Score: {report['weighted avg']['f1-score']:.4f}

Majority Class Baseline Accuracy: {baseline_accuracy:.4f}
Majority Class Baseline Balanced Accuracy: {baseline_balanced_accuracy:.4f}

Training Time: {training_time:.6f} seconds
Prediction Time: {prediction_time:.6f} seconds

Tree Depth: {tree_model.get_depth()}
Tree Leaves: {tree_model.get_n_leaves()}
"""

with open(
    OUTPUT_DIR / "decision_tree_summary.txt",
    "w",
    encoding="utf-8"
) as file:
    file.write(summary)

print(summary)

print("\n========== DECISION TREE COMPLETED ==========")
