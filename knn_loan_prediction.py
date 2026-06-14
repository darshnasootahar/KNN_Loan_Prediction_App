"""
============================================================
  K-Nearest Neighbors — Loan Approval Prediction
============================================================
"""

# ── 0. Imports ───────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings, joblib, os
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             classification_report, ConfusionMatrixDisplay)

# ── Paths (works on Windows, Mac, Linux) ─────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
CSV_PATH    = os.path.join(BASE_DIR, "train.csv")
OUTPUT_DIR  = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEED = 42

# ═══════════════════════════════════════════════════════════
# TASK 1 — DATA PREPROCESSING
# ═══════════════════════════════════════════════════════════
print("=" * 60)
print("  TASK 1 — DATA PREPROCESSING")
print("=" * 60)

# 1-A. Load & Inspect
df = pd.read_csv(CSV_PATH)
print("\n📌 First 5 rows:")
print(df.head().to_string())
print(f"\n📐 Shape: {df.shape[0]} rows × {df.shape[1]} columns")

# 1-B. Missing Values Report
print("\n📌 Missing values BEFORE imputation:")
missing     = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
mv_table    = pd.DataFrame({"Missing Count": missing, "Missing %": missing_pct})
print(mv_table[mv_table["Missing Count"] > 0].to_string())

# 1-C. Fill missing values
cat_fill = ["Gender", "Married", "Dependents", "Self_Employed"]
num_fill  = ["LoanAmount", "Loan_Amount_Term", "Credit_History"]

for col in cat_fill:
    df[col] = df[col].fillna(df[col].mode()[0])
for col in num_fill:
    df[col] = df[col].fillna(df[col].median())

print(f"\n✅ Missing values AFTER imputation: {df.isnull().sum().sum()} (none remaining)")

# 1-D. Encode categorical variables
df = df.drop(columns=["Loan_ID"])

encode_cols = ["Gender", "Married", "Dependents", "Education",
               "Self_Employed", "Property_Area", "Loan_Status"]

le = LabelEncoder()
for col in encode_cols:
    df[col] = le.fit_transform(df[col].astype(str))

print("\n📌 Encoded DataFrame (first 5 rows):")
print(df.head().to_string())

# 1-E. Train / Test Split (80 / 20)
X = df.drop(columns=["Loan_Status"])
y = df["Loan_Status"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y)

print(f"\n📌 Split → Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")

# 1-F. Standardise features
scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

joblib.dump(scaler,          os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(list(X.columns), os.path.join(OUTPUT_DIR, "feature_cols.pkl"))
print("✅ Features standardised & scaler saved.")


# ═══════════════════════════════════════════════════════════
# TASK 2 — TRAIN KNN (default + various k)
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TASK 2 — TRAIN KNN MODEL")
print("=" * 60)

knn_default = KNeighborsClassifier()
knn_default.fit(X_train_sc, y_train)
default_acc = accuracy_score(y_test, knn_default.predict(X_test_sc))
print(f"\n📌 Default KNN (k=5)  →  Test Accuracy: {default_acc:.4f}")

print("\n📌 Accuracy for different values of k:")
k_values  = [1, 3, 5, 7, 9, 11, 15, 21]
k_results = {}
for k in k_values:
    m   = KNeighborsClassifier(n_neighbors=k)
    m.fit(X_train_sc, y_train)
    acc = accuracy_score(y_test, m.predict(X_test_sc))
    k_results[k] = acc
    print(f"   k={k:>2}  →  Accuracy: {acc:.4f}")

# k-accuracy plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(list(k_results.keys()), list(k_results.values()),
        marker="o", linewidth=2.5, color="#4C6EF5", markersize=8)
ax.axvline(x=5, color="#FA5252", linestyle="--", linewidth=1.5, label="Default k=5")
ax.set_xlabel("k (Number of Neighbors)", fontsize=13)
ax.set_ylabel("Test Accuracy", fontsize=13)
ax.set_title("KNN Accuracy vs. Number of Neighbors (k)", fontsize=15, fontweight="bold")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.35)
ax.set_xticks(list(k_results.keys()))
fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "k_accuracy_plot.png"), dpi=150)
plt.close()
print("\n📊 k-accuracy plot saved.")


# ═══════════════════════════════════════════════════════════
# TASK 3 — HYPERPARAMETER TUNING
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TASK 3 — HYPERPARAMETER TUNING (GridSearchCV)")
print("=" * 60)

param_grid = {
    "n_neighbors": list(range(1, 26)),
    "weights":     ["uniform", "distance"],
    "metric":      ["euclidean", "manhattan"]
}

grid_search = GridSearchCV(
    KNeighborsClassifier(), param_grid,
    cv=5, scoring="accuracy", n_jobs=-1, verbose=0)
grid_search.fit(X_train_sc, y_train)

best_params = grid_search.best_params_
best_cv_acc = grid_search.best_score_
print(f"\n✅ Best Parameters : {best_params}")
print(f"✅ Best CV Accuracy : {best_cv_acc:.4f}")


# ═══════════════════════════════════════════════════════════
# TASK 4 — MODEL EVALUATION
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TASK 4 — MODEL EVALUATION")
print("=" * 60)

# Default KNN
y_pred_def = knn_default.predict(X_test_sc)
acc_def    = accuracy_score(y_test, y_pred_def)
cm_def     = confusion_matrix(y_test, y_pred_def)

print("\n─── Default KNN (k=5) ───────────────────────────────")
print(f"  Accuracy       : {acc_def:.4f}")
print("  Confusion Matrix:\n", cm_def)
print("\n  Classification Report:")
print(classification_report(y_test, y_pred_def,
                            target_names=["Rejected", "Approved"]))

# Tuned KNN
best_knn    = grid_search.best_estimator_
y_pred_best = best_knn.predict(X_test_sc)
acc_best    = accuracy_score(y_test, y_pred_best)
cm_best     = confusion_matrix(y_test, y_pred_best)

print(f"─── Tuned KNN (k={best_params['n_neighbors']}, "
      f"metric={best_params['metric']}, weights={best_params['weights']}) ───")
print(f"  Accuracy       : {acc_best:.4f}")
print("  Confusion Matrix:\n", cm_best)
print("\n  Classification Report:")
print(classification_report(y_test, y_pred_best,
                            target_names=["Rejected", "Approved"]))

cv_scores = cross_val_score(best_knn, X_train_sc, y_train, cv=5, scoring="accuracy")
print(f"  5-Fold CV Accuracy : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# ═══════════════════════════════════════════════════════════
# TASK 5 — VISUALISATION
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("  TASK 5 — CONFUSION MATRIX VISUALISATION")
print("=" * 60)

labels = ["Rejected", "Approved"]
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Confusion Matrices — KNN Loan Approval Prediction",
             fontsize=15, fontweight="bold", y=1.02)

for ax, cm, title, cmap in zip(
        axes,
        [cm_def, cm_best],
        [f"Default KNN  (k=5)\nAccuracy: {acc_def:.4f}",
         f"Tuned KNN  (k={best_params['n_neighbors']})\nAccuracy: {acc_best:.4f}"],
        ["Blues", "Greens"]):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap=cmap, colorbar=False)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_ylabel("True Label", fontsize=11)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
plt.close()
print("📊 Confusion matrix plots saved.")

# Save model
joblib.dump(best_knn, os.path.join(OUTPUT_DIR, "knn_model.pkl"))
print("\n✅ Best KNN model saved → outputs/knn_model.pkl")
print(f"\n✅ All output files saved to: {OUTPUT_DIR}")
print("\n🎉 All tasks complete!")
