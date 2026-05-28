import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================
# LOAD DATA
# =========================

assessments          = pd.read_csv("data/assessments.csv")
courses              = pd.read_csv("data/courses.csv")
student_assessments  = pd.read_csv("data/studentAssessments.csv")
student_info         = pd.read_csv("data/studentInfo.csv")
student_registration = pd.read_csv("data/studentRegistration.csv")
student_vle          = pd.read_csv("data/studentVle.csv")

for df in [assessments, courses, student_assessments,
           student_info, student_registration, student_vle]:
    df.replace("?", np.nan, inplace=True)

# =========================
# VLE FEATURE ENGINEERING
# =========================

vle_agg = student_vle.groupby(
    ['id_student', 'code_module', 'code_presentation']
).agg(
    total_clicks      = ('sum_click', 'sum'),
    avg_clicks        = ('sum_click', 'mean'),
    interaction_count = ('sum_click', 'count')
).reset_index()

# =========================
# MERGE
# =========================

df = student_assessments.merge(assessments,           on='id_assessment',                                    how='left')
df = df.merge(student_info,        on=['id_student','code_module','code_presentation'], how='left')
df = df.merge(student_registration,on=['id_student','code_module','code_presentation'], how='left')
df = df.merge(vle_agg,             on=['id_student','code_module','code_presentation'], how='left')
df = df.merge(courses,             on=['code_module','code_presentation'],              how='left')

print("Merged shape:", df.shape)
print("Columns:", df.columns.tolist())

# =========================
# DROP ROWS WITH NO TARGET
# =========================

df = df.dropna(subset=['assessmentClass'])
df['assessmentClass'] = df['assessmentClass'].str.strip().str.title()

print("\nClass distribution:")
print(df['assessmentClass'].value_counts())

# =========================
# FEATURE ENGINEERING
# =========================

df['date_unregistration'] = pd.to_numeric(df['date_unregistration'], errors='coerce')
df['date_registration']   = pd.to_numeric(df['date_registration'],   errors='coerce')
df['unregistered_flag']   = df['date_unregistration'].notna().astype(int)
df['days_enrolled'] = (
    df['date_unregistration'].fillna(df['date_unregistration'].median())
    - df['date_registration'].fillna(df['date_registration'].median())
)
df['weight'] = pd.to_numeric(df['weight'], errors='coerce')

# =========================
# ENCODE TARGET
# Ordinal: Fail=0, Good=1, Very Good=2, Excellent=3
# =========================

CLASS_ORDER = ['Fail', 'Good', 'Very Good', 'Excellent']
le = LabelEncoder()
le.fit(CLASS_ORDER)
df['target'] = le.transform(df['assessmentClass'])

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

DROP_COLS = ['id_student', 'id_assessment', 'assessmentClass', 'target']

# =========================
# PREPROCESSING — TRAIN
# =========================

def preprocess_train(df):
    df = df.copy()

    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    fill_values = {}
    skip_cols = {'assessmentClass', 'target'}

    for col in df.columns:
        if col in skip_cols:
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

    joblib.dump(fill_values, "fill_values_cls.pkl")
    print("\nSaved fill_values_cls.pkl")

    y = df['target'].copy()
    df.drop(columns=DROP_COLS, inplace=True, errors='ignore')
    df = pd.get_dummies(df, drop_first=True)

    return df, y


processed_df, y = preprocess_train(df)
X = processed_df

print(f"\nFinal feature count : {X.shape[1]}")
print(f"Total samples       : {X.shape[0]}")

# =========================
# TRAIN / TEST SPLIT  80 / 20
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nTrain size : {len(X_train)}")
print(f"Test size  : {len(X_test)}")

# =========================
# SCALER — fit on train only
# needed for Logistic Regression and KNN
# =========================

scaler_cls = StandardScaler()
X_train_sc = scaler_cls.fit_transform(X_train)
X_test_sc  = scaler_cls.transform(X_test)

# Save shared artifacts
joblib.dump(X.columns.tolist(), "features_cls.pkl")
joblib.dump(scaler_cls,         "scaler_cls.pkl")
joblib.dump(le,                 "label_encoder.pkl")

# =========================
# HELPER — train, evaluate, time
# =========================

def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    t0 = time.time()
    preds = model.predict(X_te)
    test_time = time.time() - t0

    acc = accuracy_score(y_te, preds)

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"  Accuracy   : {acc:.4f}")
    print(f"  Train time : {train_time:.2f}s")
    print(f"  Test time  : {test_time:.4f}s")
    print(f"{'='*50}")
    print(classification_report(y_te, preds, target_names=le.classes_))

    return acc, train_time, test_time, preds

# =========================
# MODEL 1 — RANDOM FOREST
# Tree-based bagging ensemble, no scaling needed
# Fast with n_jobs=-1 (uses all CPU cores)
# =========================

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,        # was 15
    min_samples_leaf=5,  # new — prevents overfitting
    random_state=42,
    n_jobs=-1
)
rf_acc, rf_train_t, rf_test_t, rf_preds = evaluate(
    "Random Forest", rf, X_train, y_train, X_test, y_test
)

# =========================
# MODEL 2 — LOGISTIC REGRESSION
# Linear probabilistic classifier, very fast on large datasets
# Completely different approach from tree models (linear decision boundary)
# Needs scaled features
# =========================

lr = LogisticRegression(
    max_iter=1000,
    solver='saga',       # fastest solver for large datasets
    n_jobs=-1,
    random_state=42
)
lr_acc, lr_train_t, lr_test_t, lr_preds = evaluate(
    "Logistic Regression", lr, X_train_sc, y_train, X_test_sc, y_test
)

# =========================
# MODEL 3 — K-NEAREST NEIGHBORS
# Instance-based, non-parametric — no training phase, different from both above
# Needs scaled features
# Use n_jobs=-1 to parallelize distance computation
# =========================

knn = KNeighborsClassifier(
    n_neighbors=11,
    metric='euclidean',
    n_jobs=-1
)
knn_acc, knn_train_t, knn_test_t, knn_preds = evaluate(
    "K-Nearest Neighbors", knn, X_train_sc, y_train, X_test_sc, y_test
)

# =========================
# SAVE ALL MODELS
# =========================

joblib.dump(rf,  "rf_cls_model.pkl")
joblib.dump(lr,  "lr_cls_model.pkl")
joblib.dump(knn, "knn_cls_model.pkl")
print("\nAll 3 classifiers saved.")

# =========================
# HYPERPARAMETER TUNING
# HP1: n_estimators (RF)  — 3 values, max_depth fixed = 15
# HP2: max_depth (RF)     — 3 values, n_estimators fixed = 200
# HP3: C (LR)             — 3 values, solver fixed
# HP4: n_neighbors (KNN)  — 3 values, metric fixed
# =========================

print("\n\n========== HYPERPARAMETER TUNING ==========")

# --- RF: n_estimators ---
print("\n[RF] Varying n_estimators  |  max_depth fixed = 15")
rf_ne_vals = [50, 100, 200]
rf_ne_accs = []
for n in rf_ne_vals:
    m = RandomForestClassifier(n_estimators=n, max_depth=15, random_state=42, n_jobs=-1)
    m.fit(X_train, y_train)
    a = accuracy_score(y_test, m.predict(X_test))
    rf_ne_accs.append(a)
    print(f"  n_estimators = {n:4d}  ->  Accuracy = {a:.4f}")

# --- RF: max_depth ---
print("\n[RF] Varying max_depth  |  n_estimators fixed = 200")
rf_d_vals = [5, 10, 15]
rf_d_accs = []
for d in rf_d_vals:
    m = RandomForestClassifier(n_estimators=200, max_depth=d, random_state=42, n_jobs=-1)
    m.fit(X_train, y_train)
    a = accuracy_score(y_test, m.predict(X_test))
    rf_d_accs.append(a)
    print(f"  max_depth    = {d:3d}  ->  Accuracy = {a:.4f}")

# --- LR: C (regularization strength) ---
print("\n[LR] Varying C  |  solver=saga fixed")
lr_c_vals = [0.01, 0.1, 1.0]
lr_c_accs = []
for c in lr_c_vals:
    m = LogisticRegression(C=c, max_iter=1000, solver='saga', n_jobs=-1, random_state=42)
    m.fit(X_train_sc, y_train)
    a = accuracy_score(y_test, m.predict(X_test_sc))
    lr_c_accs.append(a)
    print(f"  C            = {c:.2f}   ->  Accuracy = {a:.4f}")

# --- KNN: n_neighbors ---
print("\n[KNN] Varying n_neighbors  |  metric=euclidean fixed")
knn_k_vals = [5, 11, 21]
knn_k_accs = []
for k in knn_k_vals:
    m = KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1)
    m.fit(X_train_sc, y_train)
    a = accuracy_score(y_test, m.predict(X_test_sc))
    knn_k_accs.append(a)
    print(f"  n_neighbors  = {k:3d}    ->  Accuracy = {a:.4f}")

# =========================
# PLOTS
# =========================

model_names = ["Random Forest", "Logistic Reg.", "KNN"]
accuracies  = [rf_acc,    lr_acc,    knn_acc]
train_times = [rf_train_t,lr_train_t,knn_train_t]
test_times  = [rf_test_t, lr_test_t, knn_test_t]
colors      = ['steelblue', 'seagreen', 'tomato']

# 1. Accuracy bar chart
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(model_names, accuracies, color=colors, edgecolor='black')
ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=11)
ax.set_title("Classification Accuracy — Test Set", fontsize=13)
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1.12)
plt.tight_layout()
plt.savefig("cls_accuracy.png", dpi=100)
plt.show()

# 2. Training time bar chart
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(model_names, train_times, color=colors, edgecolor='black')
ax.bar_label(bars, fmt='%.2f', padding=4, fontsize=11)
ax.set_title("Total Training Time (seconds)", fontsize=13)
ax.set_ylabel("Time (s)")
plt.tight_layout()
plt.savefig("cls_train_time.png", dpi=100)
plt.show()

# 3. Test time bar chart
fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(model_names, test_times, color=colors, edgecolor='black')
ax.bar_label(bars, fmt='%.4f', padding=4, fontsize=11)
ax.set_title("Total Test Time (seconds)", fontsize=13)
ax.set_ylabel("Time (s)")
plt.tight_layout()
plt.savefig("cls_test_time.png", dpi=100)
plt.show()

# 4. Confusion matrix — best model
best_acc   = max(rf_acc, lr_acc, knn_acc)
best_preds = rf_preds if rf_acc == best_acc else (lr_preds if lr_acc == best_acc else knn_preds)
best_name  = "Random Forest" if rf_acc == best_acc else ("Logistic Regression" if lr_acc == best_acc else "KNN")

cm = confusion_matrix(y_test, best_preds)
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_, yticklabels=le.classes_)
ax.set_title(f"Confusion Matrix — {best_name}", fontsize=13)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.tight_layout()
plt.savefig("cls_confusion_matrix.png", dpi=100)
plt.show()

# 5. HP — RF n_estimators
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([str(v) for v in rf_ne_vals], rf_ne_accs, marker='o', color='steelblue', linewidth=2)
for x, yv in zip([str(v) for v in rf_ne_vals], rf_ne_accs):
    ax.annotate(f"{yv:.4f}", (x, yv), textcoords="offset points", xytext=(0, 8), ha='center')
ax.set_title("RF — n_estimators vs Accuracy", fontsize=12)
ax.set_xlabel("n_estimators")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("hp_rf_nestimators.png", dpi=100)
plt.show()

# 6. HP — RF max_depth
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([str(v) for v in rf_d_vals], rf_d_accs, marker='o', color='steelblue', linewidth=2)
for x, yv in zip([str(v) for v in rf_d_vals], rf_d_accs):
    ax.annotate(f"{yv:.4f}", (x, yv), textcoords="offset points", xytext=(0, 8), ha='center')
ax.set_title("RF — max_depth vs Accuracy", fontsize=12)
ax.set_xlabel("max_depth")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("hp_rf_maxdepth.png", dpi=100)
plt.show()

# 7. HP — LR C
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([str(v) for v in lr_c_vals], lr_c_accs, marker='o', color='seagreen', linewidth=2)
for x, yv in zip([str(v) for v in lr_c_vals], lr_c_accs):
    ax.annotate(f"{yv:.4f}", (x, yv), textcoords="offset points", xytext=(0, 8), ha='center')
ax.set_title("Logistic Regression — C vs Accuracy", fontsize=12)
ax.set_xlabel("C (regularization)")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("hp_lr_C.png", dpi=100)
plt.show()

# 8. HP — KNN n_neighbors
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot([str(v) for v in knn_k_vals], knn_k_accs, marker='o', color='tomato', linewidth=2)
for x, yv in zip([str(v) for v in knn_k_vals], knn_k_accs):
    ax.annotate(f"{yv:.4f}", (x, yv), textcoords="offset points", xytext=(0, 8), ha='center')
ax.set_title("KNN — n_neighbors vs Accuracy", fontsize=12)
ax.set_xlabel("n_neighbors")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("hp_knn_k.png", dpi=100)
plt.show()

print("\nDone. All models and plots saved.")