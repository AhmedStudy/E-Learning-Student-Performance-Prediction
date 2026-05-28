"""
test_unseen.py — Milestone 2 (Classification)
Run this on discussion day with the unseen CSV files.
"""

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import accuracy_score, classification_report

# =========================
# LOAD SAVED ARTIFACTS
# =========================

rf          = joblib.load("rf_cls_model.pkl")
lr          = joblib.load("lr_cls_model.pkl")
knn         = joblib.load("knn_cls_model.pkl")
scaler_cls  = joblib.load("scaler_cls.pkl")
features    = joblib.load("features_cls.pkl")
fill_values = joblib.load("fill_values_cls.pkl")
le          = joblib.load("label_encoder.pkl")

print("All saved models and artifacts loaded.")

# =========================
# LOAD UNSEEN DATA FILES
# =========================

assessments          = pd.read_csv("test_unseen/assessments.csv")
courses              = pd.read_csv("test_unseen/courses.csv")
student_assessments  = pd.read_csv("test_unseen/studentAssessments.csv")
student_info         = pd.read_csv("test_unseen/studentInfo.csv")
student_registration = pd.read_csv("test_unseen/studentRegistration.csv")
student_vle          = pd.read_csv("test_unseen/studentVle.csv")

for df in [assessments, courses, student_assessments,
           student_info, student_registration, student_vle]:
    df.replace("?", np.nan, inplace=True)

# =========================
# VLE FEATURE ENGINEERING  (same as training)
# =========================

vle_agg = student_vle.groupby(
    ['id_student', 'code_module', 'code_presentation']
).agg(
    total_clicks      = ('sum_click', 'sum'),
    avg_clicks        = ('sum_click', 'mean'),
    interaction_count = ('sum_click', 'count')
).reset_index()

# =========================
# MERGE  (same order as training)
# =========================

df = student_assessments.merge(assessments, on='id_assessment', how='left')
df = df.merge(student_info,        on=['id_student','code_module','code_presentation'], how='left')
df = df.merge(student_registration,on=['id_student','code_module','code_presentation'], how='left')
df = df.merge(vle_agg,             on=['id_student','code_module','code_presentation'], how='left')
df = df.merge(courses,             on=['code_module','code_presentation'],  how='left')

print("Unseen data shape after merge:", df.shape)

# =========================
# FEATURE ENGINEERING  (same as training)
# =========================

df['date_unregistration'] = pd.to_numeric(df['date_unregistration'], errors='coerce') if 'date_unregistration' in df.columns else 0
df['date_registration']   = pd.to_numeric(df['date_registration'],   errors='coerce') if 'date_registration'   in df.columns else 0
df['unregistered_flag']   = df['date_unregistration'].notna().astype(int)
df['days_enrolled'] = (
    df['date_unregistration'].fillna(df['date_unregistration'].median())
    - df['date_registration'].fillna(df['date_registration'].median())
)
df['weight'] = pd.to_numeric(df['weight'], errors='coerce') if 'weight' in df.columns else 0

# =========================
# EXTRACT GROUND TRUTH IF AVAILABLE
# =========================

y_true     = None
valid_mask = None
y_true_enc = None

if 'assessmentClass' in df.columns:
    df['assessmentClass'] = df['assessmentClass'].str.strip().str.title()
    valid_mask = df['assessmentClass'].isin(le.classes_)
    if valid_mask.sum() > 0:
        y_true     = df['assessmentClass'].copy()
        y_true_enc = le.transform(y_true[valid_mask])

# =========================
# CONSTANTS
# =========================

NUMERIC_COLS = [
    'date', 'weight', 'date_registration', 'date_unregistration',
    'total_clicks', 'avg_clicks', 'interaction_count',
    'module_presentation_length', 'studied_credits',
    'num_of_prev_attempts', 'days_enrolled', 'unregistered_flag',
    'is_banked', 'date_submitted'
]

# Drop target and ID columns — must match training exactly
DROP_COLS = ['id_student', 'id_assessment', 'assessmentClass', 'target']

# =========================
# PREPROCESSING — training fill values only, no refitting
# =========================

for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

skip_cols = {'assessmentClass', 'target'}
for col in df.columns:
    if col in skip_cols:
        continue
    if col in fill_values:
        _, fill_val = fill_values[col]
        df[col] = df[col].fillna(fill_val)
    else:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("Unknown")

df.drop(columns=DROP_COLS, inplace=True, errors='ignore')
df = pd.get_dummies(df, drop_first=True)

# Align to EXACT training feature set
for col in features:
    if col not in df.columns:
        df[col] = 0
df = df[features]

print(f"Feature count after alignment: {df.shape[1]}")

# =========================
# SCALE for LR and KNN
# =========================

df_scaled = scaler_cls.transform(df)

# =========================
# PREDICT — ALL 3 MODELS
# =========================

rf_preds  = rf.predict(df)
lr_preds  = lr.predict(df_scaled)
knn_preds = knn.predict(df_scaled)

rf_labels  = le.inverse_transform(rf_preds)
lr_labels  = le.inverse_transform(lr_preds)
knn_labels = le.inverse_transform(knn_preds)

# =========================
# EVALUATE IF GROUND TRUTH EXISTS
# =========================

if y_true is not None and valid_mask.sum() > 0:
    print("\n========== CLASSIFICATION ACCURACY ON UNSEEN DATA ==========")

    rf_acc  = accuracy_score(y_true_enc, rf_preds[valid_mask])
    lr_acc  = accuracy_score(y_true_enc, lr_preds[valid_mask])
    knn_acc = accuracy_score(y_true_enc, knn_preds[valid_mask])

    print(f"\nRandom Forest        Accuracy : {rf_acc:.4f}")
    print(f"Logistic Regression  Accuracy : {lr_acc:.4f}")
    print(f"KNN                  Accuracy : {knn_acc:.4f}")

    print("\n--- Random Forest Report ---")
    print(classification_report(y_true_enc, rf_preds[valid_mask], target_names=le.classes_))

    print("\n--- Logistic Regression Report ---")
    print(classification_report(y_true_enc, lr_preds[valid_mask], target_names=le.classes_))

    print("\n--- KNN Report ---")
    print(classification_report(y_true_enc, knn_preds[valid_mask], target_names=le.classes_))

else:
    print("\nNo ground-truth 'assessmentClass' found — showing predictions only.")

# =========================
# SAVE OUTPUT
# =========================

out = pd.DataFrame({
    "rf_predicted_class" : rf_labels,
    "lr_predicted_class" : lr_labels,
    "knn_predicted_class": knn_labels
})

if y_true is not None:
    out.insert(0, "actual_class", y_true.values)

out.to_csv("unseen_predictions_classification.csv", index=False)
print("\nPredictions saved -> unseen_predictions_classification.csv")
