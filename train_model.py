"""
=============================================================
  Pocket Legal Assistant — Model Training Script (v2)
  train_model.py
=============================================================
  Upgrades over baseline:
    • TF-IDF captures bigrams  (ngram_range=(1, 2))
    • GridSearchCV tunes C ∈ [0.1, 1, 10, 100]
                       kernel ∈ ['linear', 'rbf']
      using 5-fold stratified cross-validation
    • Best estimator is printed and evaluated on test set
    • Best model + updated vectorizer overwrite old .pkl files

  Workflow:
    1. Load  →  2. Split  →  3. TF-IDF (bigrams)
    →  4. GridSearchCV  →  5. Evaluate  →  6. Save
=============================================================
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import time

# ─────────────────────────────────────────────
# 1.  LOAD DATASET
# ─────────────────────────────────────────────
DATASET_PATH = "expanded_legal_dataset.csv"

print("=" * 65)
print("  Pocket Legal Assistant — Model Trainer  (v2 · GridSearchCV)")
print("=" * 65)

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(
        f"Dataset not found at '{DATASET_PATH}'. "
        "Please make sure the CSV is in the same directory."
    )

df = pd.read_csv(DATASET_PATH)

# Basic sanity checks
assert "User_Scenario"  in df.columns, "Missing column: User_Scenario"
assert "Legal_Category" in df.columns, "Missing column: Legal_Category"

# Drop any rows where either column is NaN
df.dropna(subset=["User_Scenario", "Legal_Category"], inplace=True)

print(f"\n✅  Dataset loaded successfully.")
print(f"    Total samples : {len(df)}")
print(f"    Categories    : {sorted(df['Legal_Category'].unique())}\n")

# ─────────────────────────────────────────────
# 2.  TRAIN / TEST SPLIT  (80 / 20, stratified)
# ─────────────────────────────────────────────
X = df["User_Scenario"]
y = df["Legal_Category"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y          # preserve class distribution in both splits
)

print(f"✅  Data split complete.")
print(f"    Training samples : {len(X_train)}")
print(f"    Testing  samples : {len(X_test)}\n")

# ─────────────────────────────────────────────
# 3.  TF-IDF VECTORISATION  (unigrams + bigrams)
#
#     ngram_range=(1, 2) → the model learns from individual words
#     AND two-word phrases, capturing context that matters in legal
#     text e.g. "security deposit", "cyber cell", "legal notice".
# ─────────────────────────────────────────────
vectorizer = TfidfVectorizer(
    stop_words="english",       # remove common English stop-words
    ngram_range=(1, 2),         # unigrams + bigrams
    max_features=10_000,        # cap vocabulary size for efficiency
    sublinear_tf=True           # apply log normalisation to term frequencies
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec  = vectorizer.transform(X_test)

print(f"✅  TF-IDF vectorisation complete.")
print(f"    Vocabulary size : {len(vectorizer.vocabulary_):,}  "
      f"(unigrams + bigrams)\n")

# ─────────────────────────────────────────────
# 4.  HYPERPARAMETER TUNING WITH GridSearchCV
#
#     Parameter grid:
#       C      : regularisation strength → [0.1, 1, 10, 100]
#                  lower C = wider margin (more regularisation)
#                  higher C = harder margin (fits training data closer)
#       kernel : decision boundary shape → ['linear', 'rbf']
#                  linear  = fast, great for high-dim text features
#                  rbf     = handles non-linear boundaries
#
#     cv=5    → 5-fold stratified cross-validation on the training set
#     n_jobs  → use all available CPU cores to parallelise the search
#     refit   → after search, refit best params on the full training set
# ─────────────────────────────────────────────
param_grid = {
    "C":      [0.1, 1, 10, 100],
    "kernel": ["linear", "rbf"],
}

base_svc = SVC(
    class_weight="balanced",    # handles any class imbalance
    probability=True,           # enables predict_proba (useful for future extensions)
    random_state=42,
)

grid_search = GridSearchCV(
    estimator=base_svc,
    param_grid=param_grid,
    cv=5,                       # 5-fold cross-validation
    scoring="accuracy",         # optimise for overall accuracy
    n_jobs=-1,                  # parallelise across all CPU cores
    verbose=2,                  # show progress for each fold
    refit=True,                 # refit best model on full X_train_vec
)

n_combos = len(param_grid["C"]) * len(param_grid["kernel"])
print(f"⏳  Starting GridSearchCV …")
print(f"    Parameter combinations : {n_combos}  "
      f"({len(param_grid['C'])} C values × {len(param_grid['kernel'])} kernels)")
print(f"    Folds per combination  : 5")
print(f"    Total fits             : {n_combos * 5}\n")

t_start = time.time()
grid_search.fit(X_train_vec, y_train)
elapsed = time.time() - t_start

print(f"\n✅  Grid search complete in {elapsed:.1f}s.\n")

# ─────────────────────────────────────────────
# PRINT BEST HYPERPARAMETERS
# ─────────────────────────────────────────────
print("=" * 65)
print("  BEST HYPERPARAMETERS")
print("=" * 65)
print(f"\n  best_params_  : {grid_search.best_params_}")
print(f"  Best CV Score : {grid_search.best_score_ * 100:.2f}%  "
      f"(mean accuracy across 5 folds)\n")

# ─────────────────────────────────────────────
# 5.  EVALUATE BEST MODEL ON THE HELD-OUT TEST SET
# ─────────────────────────────────────────────
best_model = grid_search.best_estimator_   # already refit on full X_train_vec

y_pred   = best_model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 65)
print("  EVALUATION RESULTS  (best model on held-out test set)")
print("=" * 65)
print(f"\n  Accuracy Score : {accuracy * 100:.2f}%\n")
print("  Classification Report:")
print("  " + "-" * 59)
print(classification_report(y_test, y_pred, zero_division=0))
print("=" * 65)

# ─────────────────────────────────────────────
# FULL GRID SEARCH SUMMARY  (all combinations, ranked)
# ─────────────────────────────────────────────
print("\n  All parameter combinations ranked by CV accuracy:")
print("  " + "-" * 59)

results = pd.DataFrame(grid_search.cv_results_)
ranked = (
    results[["param_C", "param_kernel", "mean_test_score", "std_test_score"]]
    .sort_values("mean_test_score", ascending=False)
    .reset_index(drop=True)
)
ranked["mean_test_score"] = (ranked["mean_test_score"] * 100).map("{:.2f}%".format)
ranked["std_test_score"]  = (ranked["std_test_score"]  * 100).map("±{:.2f}%".format)
ranked.index += 1
ranked.columns = ["C", "kernel", "Mean CV Acc", "Std Dev"]
print(ranked.to_string())
print()

# ─────────────────────────────────────────────
# 6.  SAVE ARTEFACTS  (overwrite old .pkl files)
# ─────────────────────────────────────────────
MODEL_PATH      = "legal_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

joblib.dump(best_model, MODEL_PATH)
joblib.dump(vectorizer, VECTORIZER_PATH)

print(f"✅  Best model saved      → {MODEL_PATH}")
print(f"✅  Vectorizer saved      → {VECTORIZER_PATH}")
print("\n🎉  Done!  The Streamlit app will automatically use the new model.\n")
