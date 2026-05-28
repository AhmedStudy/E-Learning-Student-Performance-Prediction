"""
test_unseen.py  —  Milestone 1 (Regression)
Run this on discussion day with the unseen CSV files.

Usage:
    python test_unseen.py
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from sklearn.metrics import r2_score, mean_squared_error

# =========================
# LOAD SAVED ARTIFACTS
# =========================

gbr               = joblib.load("gbr_model.pkl")
ridge             = joblib.load("ridge_model.pkl")
scaler            = joblib.load("scaler.pkl")
features          = joblib.load("features.pkl")
fill_values       = joblib.load("fill_values.pkl")
global_mean_score = joblib.load("global_mean_score.pkl")
module_weight_df  = joblib.load("module_total_weight.pkl")
p99_clicks        = joblib.load("clicks_per_day_p99.pkl")
assessment_stats  = joblib.load("assessment_stats.pkl")   # ← training-derived per-assessment stats

# =========================
# LOAD UNSEEN DATA FILES
# =========================

assessments          = pd.read_csv("test_unseen/assessments.csv")
courses              = pd.read_csv("test_unseen/courses.csv")
student_assessments  = pd.read_csv("test_unseen/studentAssessments.csv")
student_info         = pd.read_csv("test_unseen/studentInfo.csv")
student_registration = pd.read_csv("test_unseen/studentRegistration.csv")
student_vle          = pd.read_csv("test_unseen/studentVle.csv")

for frame in [assessments, courses, student_assessments,
              student_info, student_registration, student_vle]:
    frame.replace("?", np.nan, inplace=True)

# Cast numeric columns
student_vle['sum_click']                    = pd.to_numeric(student_vle['sum_click'],                    errors='coerce')
student_assessments['score']                = pd.to_numeric(student_assessments['score'],                errors='coerce')
student_assessments['date_submitted']       = pd.to_numeric(student_assessments['date_submitted'],       errors='coerce')
student_assessments['is_banked']            = pd.to_numeric(student_assessments['is_banked'],            errors='coerce')
student_registration['date_registration']   = pd.to_numeric(student_registration['date_registration'],   errors='coerce')
student_registration['date_unregistration'] = pd.to_numeric(student_registration['date_unregistration'], errors='coerce')
assessments['date']                         = pd.to_numeric(assessments['date'],                         errors='coerce')
assessments['weight']                       = pd.to_numeric(assessments['weight'],                       errors='coerce')
courses['module_presentation_length']       = pd.to_numeric(courses['module_presentation_length'],       errors='coerce')
student_info['num_of_prev_attempts']        = pd.to_numeric(student_info['num_of_prev_attempts'],        errors='coerce')
student_info['studied_credits']             = pd.to_numeric(student_info['studied_credits'],             errors='coerce')

# =========================
# STEP 1: Assessment difficulty stats
# Use TRAINING-DERIVED stats (saved as assessment_stats.pkl).
# Do NOT recompute from unseen scores — that would leak test labels.
# Drop any columns that may already exist in the unseen assessments.csv
# to avoid conflicts, then merge the training stats in.
# =========================

assessments = assessments.drop(
    columns=[c for c in ['assessment_avg_score', 'assessment_score_std'] if c in assessments.columns]
)
assessments = assessments.merge(assessment_stats, on='id_assessment', how='left')
assessments['assessment_avg_score'] = assessments['assessment_avg_score'].fillna(global_mean_score)
assessments['assessment_score_std'] = assessments['assessment_score_std'].fillna(0.0)

# =========================
# STEP 2: Attach code_module/code_presentation to student_assessments
# =========================

sa = student_assessments.merge(
    assessments[['id_assessment', 'code_module', 'code_presentation']],
    on='id_assessment', how='left'
)

# =========================
# STEP 3: Student historical performance  ← THE KEY FIX
#
# In training we used expanding().mean() with shift(1) so each row only
# sees scores from the student's *earlier* submissions in the same module.
# We must do the EXACT same thing here on the unseen data — the unseen
# dataset has the same row structure, and the scores are present (we just
# hide them from the model by not using raw score as a feature).
#
# Hardcoding global_mean_score for every row (the old approach) made this
# feature look completely different at test time vs. train time, which is
# why R² collapsed to ~0.08. This fix restores the proper distribution.
# =========================

sa = sa.sort_values(['id_student', 'code_module', 'code_presentation', 'date_submitted']).copy()

sa['student_hist_mean_score'] = (
    sa.groupby(['id_student', 'code_module', 'code_presentation'])['score']
    .transform(lambda x: x.shift(1).expanding().mean())
)
sa['student_hist_count'] = (
    sa.groupby(['id_student', 'code_module', 'code_presentation'])['score']
    .transform(lambda x: x.shift(1).expanding().count())
)

# First assessment per student has no history → fall back to population mean
sa['student_hist_mean_score'] = sa['student_hist_mean_score'].fillna(global_mean_score)
sa['student_hist_count']      = sa['student_hist_count'].fillna(0)

# Drop helper columns before full merge (they come back from assessments merge)
sa = sa.drop(columns=['code_module', 'code_presentation'])

# =========================
# STEP 4: VLE aggregates
# =========================

vle_agg = student_vle.groupby(
    ['id_student', 'code_module', 'code_presentation']
).agg(
    total_clicks      = ('sum_click', 'sum'),
    avg_clicks        = ('sum_click', 'mean'),
    interaction_count = ('sum_click', 'count'),
    max_clicks_day    = ('sum_click', 'max'),
    std_clicks        = ('sum_click', 'std'),
).reset_index()
vle_agg['std_clicks'] = vle_agg['std_clicks'].fillna(0)

# =========================
# STEP 5: Full merge (same order as training)
# =========================

df = sa.merge(assessments,            on='id_assessment',                                    how='left')
df = df.merge(student_info,         on=['id_student','code_module','code_presentation'], how='left')
df = df.merge(student_registration, on=['id_student','code_module','code_presentation'], how='left')
df = df.merge(vle_agg,              on=['id_student','code_module','code_presentation'], how='left')
df = df.merge(courses,              on=['code_module','code_presentation'],              how='left')

print("Unseen data shape after merge:", df.shape)

# =========================
# SEPARATE GROUND TRUTH
# =========================

y_true_final = None
if 'score' in df.columns:
    df['score'] = pd.to_numeric(df['score'], errors='coerce')
    y_true_final = df['score'].copy()
    df = df.drop(columns=['score'])

# =========================
# POST-MERGE FEATURE ENGINEERING  (mirrors training exactly)
# =========================

df['date_unregistration'] = pd.to_numeric(df['date_unregistration'], errors='coerce')
df['date_registration']   = pd.to_numeric(df['date_registration'],   errors='coerce')

df['unregistered_flag'] = df['date_unregistration'].notna().astype(int)
df['days_enrolled'] = (
    df['date_unregistration'].fillna(df['date_unregistration'].median())
    - df['date_registration'].fillna(df['date_registration'].median())
)

df['weight'] = pd.to_numeric(df['weight'], errors='coerce')
df['days_early']     = df['date'] - df['date_submitted']
df['submitted_late'] = (df['days_early'] < 0).astype(int)

df['clicks_per_day'] = df['total_clicks'] / (df['days_enrolled'].replace(0, np.nan))
df['clicks_per_day'] = df['clicks_per_day'].fillna(0).clip(upper=p99_clicks)  # training p99

# Use TRAINING module totals — do not refit on unseen data
df = df.merge(module_weight_df, on='code_module', how='left')
df['relative_weight'] = df['weight'] / df['module_total_weight'].replace(0, np.nan)
df['relative_weight'] = df['relative_weight'].fillna(0)

# =========================
# DROP LEAKING COLUMNS
# =========================

leaking_cols = ['final_result', 'weighted_score']
df.drop(columns=[c for c in leaking_cols if c in df.columns], inplace=True)

# =========================
# CONSTANTS
# =========================

NUMERIC_COLS = [
    'date', 'weight', 'date_registration', 'date_unregistration',
    'total_clicks', 'avg_clicks', 'interaction_count', 'max_clicks_day', 'std_clicks',
    'module_presentation_length', 'studied_credits', 'num_of_prev_attempts',
    'days_enrolled', 'unregistered_flag', 'days_early', 'submitted_late',
    'clicks_per_day', 'relative_weight', 'module_total_weight',
    'assessment_avg_score', 'assessment_score_std',
    'student_hist_mean_score', 'student_hist_count',
    'is_banked', 'date_submitted'
]

DROP_COLS = ['id_student', 'id_assessment']

# =========================
# PREPROCESSING — training fill values only, no refitting
# =========================

for col in NUMERIC_COLS:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

for col in df.columns:
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

print("Feature count after alignment:", df.shape[1])

# =========================
# PREDICT
# =========================

gbr_preds   = gbr.predict(df)
df_scaled   = scaler.transform(df)
ridge_preds = ridge.predict(df_scaled)

# =========================
# EVALUATE
# =========================

if y_true_final is not None and y_true_final.notna().sum() > 0:
    mask = y_true_final.notna()
    print("\n========== EVALUATION ON UNSEEN DATA ==========")
    print(f"\nHist Gradient Boosting:")
    print(f"  MSE : {mean_squared_error(y_true_final[mask], gbr_preds[mask]):.4f}")
    print(f"  R²  : {r2_score(y_true_final[mask],           gbr_preds[mask]):.4f}")
    print(f"\nRidge Regression:")
    print(f"  MSE : {mean_squared_error(y_true_final[mask], ridge_preds[mask]):.4f}")
    print(f"  R²  : {r2_score(y_true_final[mask],           ridge_preds[mask]):.4f}")
else:
    print("\nNo ground-truth 'score' column found — showing predictions only.")

# =========================
# SAVE OUTPUT
# =========================

out = pd.DataFrame({
    "gbr_predicted_score"  : gbr_preds,
    "ridge_predicted_score": ridge_preds
})

if y_true_final is not None:
    out.insert(0, "actual_score", y_true_final.values)

out.to_csv("unseen_predictions_regression.csv", index=False)
print("\nPredictions saved → unseen_predictions_regression.csv")
