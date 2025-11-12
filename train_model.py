# train_model.py
import json
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

# CONFIG
DATA_PATH = Path("Expresso_churn_dataset.csv")
MODEL_PATH = Path("model.joblib")
METADATA_PATH = Path("metadata.json")
SAMPLE_SIZE = 200_000  
RANDOM_STATE = 42


assert DATA_PATH.exists(), f"Dataset not found at {DATA_PATH}"

# 1) Load dataframe
df = pd.read_csv(DATA_PATH)

# 2) Quick checks
print("Data shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Missing counts (top 20):")
print(df.isna().sum().sort_values(ascending=False).head(20))

# 3) Select features and target
# We'll drop user_id and use the rest (except CHURN is target)
TARGET = "CHURN"
DROP_COLS = ["user_id"]
features = [c for c in df.columns if c not in DROP_COLS + [TARGET]]

# 4) sample to speed up training
if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(df):
    df_sample = df.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)
else:
    df_sample = df.copy()

# 5) Prepare X and y
X = df_sample[features].copy()
y = df_sample[TARGET].astype(int).copy()

# 6) Determine numeric and categorical features
numeric_feats = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_feats = X.select_dtypes(include=["object", "category"]).columns.tolist()

print("Numeric features:", numeric_feats)
print("Categorical features:", categorical_feats)

# 7) Fix obvious dtype issues (e.g., TENURE might be object but numeric-like)
# Try to coerce numeric-like object columns to numeric (safe attempt)
for col in categorical_feats[:]:
    if X[col].dropna().str.replace(".", "", 1).str.isnumeric().all():
        X[col] = pd.to_numeric(X[col], errors='coerce')
        if X[col].dtype != object:
            categorical_feats.remove(col)
            numeric_feats.append(col)

# 8) Build preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="__missing__")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_feats),
        ("cat", categorical_transformer, categorical_feats),
    ],
    remainder="drop",  # drop any other columns
    sparse_threshold=0
)

# 9) Build full pipeline with classifier
clf = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1, class_weight="balanced"))
])

# 10) Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.15, random_state=RANDOM_STATE)

print("Training size:", X_train.shape, "Test size:", X_test.shape)

# 11) Fit
clf.fit(X_train, y_train)

# 12) Evaluate
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]
print("\nClassification report (test):")
print(classification_report(y_test, y_pred, digits=4))
try:
    auc = roc_auc_score(y_test, y_proba)
    print(f"\nROC AUC: {auc:.4f}")
except Exception as e:
    print("ROC AUC could not be computed:", e)

# 13) Save model
joblib.dump(clf, MODEL_PATH)
print(f"Saved model to {MODEL_PATH}")

# 14) Save metadata for Streamlit app (unique categories for categorical features)
metadata = {}
for c in categorical_feats:
    values = df[c].dropna().unique().tolist()
    # Sort values for nicer UI
    values_sorted = sorted([str(v) for v in values])
    metadata[c] = values_sorted

# Also save list of numeric feature names (to build input form)
metadata["_numeric_features"] = numeric_feats
metadata["_categorical_features"] = categorical_feats
metadata["_features_order"] = features

with open(METADATA_PATH, "w") as f:
    json.dump(metadata, f, indent=2)
print(f"Saved metadata to {METADATA_PATH}")
