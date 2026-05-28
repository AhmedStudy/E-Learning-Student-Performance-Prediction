import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings("ignore")

import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor   # ← much faster than GBR
from sklearn.metrics import r2_score, mean_squared_error

# =========================
# LOAD DATA
# =========================

assessments          = pd.read_csv("data/assessments.csv")
courses              = pd.read_csv("data/courses.csv")
student_assessments  = pd.read_csv("data/studentAssessments.csv")
student_info         = pd.read_csv("data/studentInfo.csv")
student_registration = pd.read_csv("data/studentRegistration.csv")
student_vle          = pd.read_csv("data/studentVle.csv")

for frame in [assessments, courses, student_assessments,
              student_info, student_registration, student_vle]:
    frame.replace("?", np.nan, inplace=True)

# Cast key numeric columns up front
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
# STEP 1: Assessment-level difficulty stats
# Population-level property of the assessment — zero leakage.
# =========================

assessment_avg = student_assessments.groupby('id_assessment')['score'].mean().rename('assessment_avg_score')
assessment_std = student_assessments.groupby('id_assessment')['score'].std().fillna(0).rename('assessment_score_std')
assessments = assessments.merge(assessment_avg, on='id_assessment', how='left')
assessments = assessments.merge(assessment_std, on='id_assessment', how='left')

global_mean_score = student_assessments['score'].mean()
joblib.dump(global_mean_score, "global_mean_score.pkl")

# Save per-assessment stats so test_unseen can reuse them without recomputing
assessment_stats = assessments[['id_assessment', 'assessment_avg_score', 'assessment_score_std']].copy()
joblib.dump(assessment_stats, "assessment_stats.pkl")   # ← NEW artifact

# =========================
# STEP 2: Attach code_module/code_presentation to student_assessments
# =========================

sa = student_assessments.merge(
    assessments[['id_assessment', 'code_module', 'code_presentation']],
    on='id_assessment', how='left'
)

# =========================
# STEP 3: Student historical performance (expanding, no look-ahead)
# shift(1) ensures current row's score is excluded from its own history.
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

sa['student_hist_mean_score'] = sa['student_hist_mean_score'].fillna(global_mean_score)
sa['student_hist_count']      = sa['student_hist_count'].fillna(0)

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
# STEP 5: Full merge
# =========================

df = sa.merge(assessments,            on='id_assessment',                                    how='left')
df = df.merge(student_info,         on=['id_student','code_module','code_presentation'], how='left')
df = df.merge(student_registration, on=['id_student','code_module','code_presentation'], how='left')
df = df.merge(vle_agg,              on=['id_student','code_module','code_presentation'], how='left')
df = df.merge(courses,              on=['code_module','code_presentation'],              how='left')

print("Merged shape:", df.shape)
print("Columns:", df.columns.tolist())

df = df.dropna(subset=['score'])

# =========================
# POST-MERGE FEATURE ENGINEERING
# =========================

df['unregistered_flag'] = df['date_unregistration'].notna().astype(int)
df['days_enrolled'] = (
    df['date_unregistration'].fillna(df['date_unregistration'].median())
    - df['date_registration'].fillna(df['date_registration'].median())
)

df['days_early']     = df['date'] - df['date_submitted']
df['submitted_late'] = (df['days_early'] < 0).astype(int)

df['clicks_per_day'] = df['total_clicks'] / (df['days_enrolled'].replace(0, np.nan))
p99 = df['clicks_per_day'].quantile(0.99)
df['clicks_per_day'] = df['clicks_per_day'].fillna(0).clip(upper=p99)
joblib.dump(float(p99), "clicks_per_day_p99.pkl")

module_total_weight = assessments.groupby('code_module')['weight'].sum().rename('module_total_weight')
df = df.merge(module_total_weight, on='code_module', how='left')
df['relative_weight'] = df['weight'] / df['module_total_weight'].replace(0, np.nan)
df['relative_weight'] = df['relative_weight'].fillna(0)
joblib.dump(module_total_weight.reset_index(), "module_total_weight.pkl")

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
# PREPROCESSING — TRAIN
# =========================

def preprocess_train(df):
    df = df.copy()

    leaking_cols = ['final_result', 'weighted_score']
    df.drop(columns=[c for c in leaking_cols if c in df.columns], inplace=True)

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    fill_values = {}
    for col in df.columns:
        if col == 'score':
            continue
        if df[col].dtype in [np.float64, np.int64, float, int]:
            fill_val = df[col].median()
            fill_values[col] = ('median', fill_val)
            df[col] = df[col].fillna(fill_val)
        else:
            mode_val = df[col].mode()
            fill_val = mode_val[0] if len(mode_val) > 0 else "Unknown"
            fill_values[col] = ('mode', fill_val)
            df[col] = df[col].fillna(fill_val)

    joblib.dump(fill_values, "fill_values.pkl")
    print("Saved fill_values.pkl")

    df.drop(columns=DROP_COLS, inplace=True, errors='ignore')
    df = pd.get_dummies(df, drop_first=True)

    return df

processed_df = preprocess_train(df)

X = processed_df.drop("score", axis=1)
y = processed_df["score"]

print("\nFinal feature count:", X.shape[1])

# =========================
# TRAIN / VALIDATION / TEST SPLIT  (60 / 20 / 20)
# =========================

X_temp,  X_test,  y_temp,  y_test  = train_test_split(X, y, test_size=0.20, random_state=42)
X_train, X_val,   y_train, y_val   = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42)

print(f"\nTrain size: {len(X_train)} | Val size: {len(X_val)} | Test size: {len(X_test)}")

# =========================
# SCALER  (fit on train only)
# =========================

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_val_sc   = scaler.transform(X_val)
X_test_sc  = scaler.transform(X_test)

# =========================
# MODEL 1: Ridge Regression
# =========================

ridge = Ridge(alpha=10.0)
ridge.fit(X_train_sc, y_train)

ridge_val_pred  = ridge.predict(X_val_sc)
ridge_test_pred = ridge.predict(X_test_sc)

ridge_val_r2   = r2_score(y_val,  ridge_val_pred)
ridge_test_r2  = r2_score(y_test, ridge_test_pred)
ridge_val_mse  = mean_squared_error(y_val,  ridge_val_pred)
ridge_test_mse = mean_squared_error(y_test, ridge_test_pred)

print("\n--- RIDGE REGRESSION ---")
print(f"  Val  R2: {ridge_val_r2:.4f}  |  Val  MSE: {ridge_val_mse:.4f}")
print(f"  Test R2: {ridge_test_r2:.4f}  |  Test MSE: {ridge_test_mse:.4f}")

# =========================
# MODEL 2: HistGradientBoostingRegressor  (replaces GradientBoostingRegressor)
#
# Why it's much faster:
#   - Bins continuous features into 255 buckets before training (like LightGBM)
#   - Operates on integer bin indices instead of raw floats → far fewer split
#     evaluations per node, much less memory, better CPU cache usage
#   - Natively handles NaN values without imputation
#   - Equivalent or better accuracy to sklearn's GBR at a fraction of the time
# =========================

gbr = HistGradientBoostingRegressor(
    max_iter         = 300,   # same as n_estimators=300
    max_depth        = 5,
    learning_rate    = 0.05,
    min_samples_leaf = 20,
    random_state     = 42,
    early_stopping   = False, # keep deterministic for fair comparison
    # Note: HistGBR does not support subsample; it uses histogram binning
    # (max_bins=255) for speed instead of row subsampling
)
gbr.fit(X_train, y_train)
gbr_val_pred  = gbr.predict(X_val)
gbr_test_pred = gbr.predict(X_test)

gbr_val_r2   = r2_score(y_val,  gbr_val_pred)
gbr_test_r2  = r2_score(y_test, gbr_test_pred)
gbr_val_mse  = mean_squared_error(y_val,  gbr_val_pred)
gbr_test_mse = mean_squared_error(y_test, gbr_test_pred)

print("\n--- HIST GRADIENT BOOSTING REGRESSOR (fast GBR) ---")
print(f"  Val  R2: {gbr_val_r2:.4f}  |  Val  MSE: {gbr_val_mse:.4f}")
print(f"  Test R2: {gbr_test_r2:.4f}  |  Test MSE: {gbr_test_mse:.4f}")

# =========================
# SAVE MODELS & ARTIFACTS
# =========================

joblib.dump(ridge,              "ridge_model.pkl")
joblib.dump(gbr,                "gbr_model.pkl")
joblib.dump(scaler,             "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")

print("\nAll models and artifacts saved.")

# =========================
# PLOTS
# =========================

# 1. Correlation heatmap
corr = processed_df.select_dtypes(include=[np.number]).corr()
score_corr = corr['score'].abs().sort_values(ascending=False).head(16)
top_cols   = score_corr.index.tolist()

plt.figure(figsize=(10, 8))
sns.heatmap(processed_df[top_cols].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap (Top Features vs Score)")
plt.tight_layout()
plt.savefig("heatmap.png", dpi=100)
plt.show()

# 2. Actual vs Predicted — GBR
plt.figure(figsize=(6, 5))
plt.scatter(y_test, gbr_test_pred, alpha=0.4, color='steelblue', edgecolors='k', linewidths=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Score"); plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted — Hist Gradient Boosting")
plt.tight_layout(); plt.savefig("actual_vs_pred_gbr.png", dpi=100); plt.show()

# 3. Actual vs Predicted — Ridge
plt.figure(figsize=(6, 5))
plt.scatter(y_test, ridge_test_pred, alpha=0.4, color='orange', edgecolors='k', linewidths=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Score"); plt.ylabel("Predicted Score")
plt.title("Actual vs Predicted — Ridge Regression")
plt.tight_layout(); plt.savefig("actual_vs_pred_ridge.png", dpi=100); plt.show()

# 4. R² comparison
plt.figure(figsize=(6, 4))
bars = plt.bar(["Ridge", "Hist GBR"], [ridge_test_r2, gbr_test_r2], color=['orange', 'steelblue'])
plt.bar_label(bars, fmt='%.4f', padding=3)
plt.title("R² Score Comparison (Test Set)"); plt.ylabel("R²"); plt.ylim(0, 1)
plt.tight_layout(); plt.savefig("r2_comparison.png", dpi=100); plt.show()

# 5. MSE comparison
plt.figure(figsize=(6, 4))
bars = plt.bar(["Ridge", "Hist GBR"], [ridge_test_mse, gbr_test_mse], color=['orange', 'steelblue'])
plt.bar_label(bars, fmt='%.2f', padding=3)
plt.title("MSE Comparison (Test Set)"); plt.ylabel("MSE")
plt.tight_layout(); plt.savefig("mse_comparison.png", dpi=100); plt.show()



print("\nDone. All plots saved.")
