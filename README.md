# E-Learning Student Performance Prediction

A comprehensive machine learning project developed at Ain Shams University (FCIS) using the Open University Learning Analytics Dataset (OULAD) to analyze, predict, and classify student academic performance based on demographic, behavioral, and engagement data from Virtual Learning Environments (VLEs).

The project is divided into two major phases:

* **Milestone 1:** Regression & Score Prediction
* **Milestone 2:** Performance Classification & Hyperparameter Tuning

---

# Project Objectives

The main objective of this project is to leverage machine learning techniques to:

* Predict student assessment scores
* Classify student performance levels
* Analyze learning behavior patterns
* Identify key factors affecting academic success
* Compare machine learning models and tuning strategies

---

# My Contribution

Although this project was submitted as part of a team assignment, I independently designed and implemented the complete machine learning pipeline, including:

* Data preprocessing and cleaning
* Feature engineering
* Exploratory Data Analysis (EDA)
* Regression and classification models
* Hyperparameter tuning
* Model evaluation and visualization
* Leakage prevention pipeline
* Inference pipeline for unseen data
* README documentation and project structuring

The project was developed using Python, Scikit-learn, Pandas, and modern machine learning workflows.

---

# Dataset

The project uses the **Open University Learning Analytics Dataset (OULAD)**, a large educational dataset containing anonymized student information, assessments, registration data, and VLE interaction records.

### Dataset Components

* `assessments.csv`
* `courses.csv`
* `studentAssessments.csv`
* `studentInfo.csv`
* `studentRegistration.csv`
* `studentVle.csv`

---

# Milestone 1 — Regression & Score Prediction

## Goal

Predict the exact numerical assessment score (0–100) for each student.

## Implemented Techniques

* Data preprocessing and cleaning
* Missing value handling
* Feature engineering
* Leakage prevention
* Feature scaling
* Exploratory Data Analysis (EDA)

## Engineered Features

* `student_hist_mean_score`
* `clicks_per_day`
* `days_early`
* `submitted_late`
* `relative_weight`
* `assessment_avg_score`
* `assessment_score_std`

## Regression Models

### Ridge Regression

A linear regression model with L2 regularization used as a baseline model.

### Hist Gradient Boosting Regressor

A non-linear ensemble model capable of capturing complex feature interactions and achieving superior predictive performance.

## Regression Results

| Model                  | Test R² | Test MSE |
| ---------------------- | ------- | -------- |
| Ridge Regression       | 0.3670  | 226.4412 |
| Hist Gradient Boosting | 0.4487  | 197.2215 |

---

# Milestone 2 — Classification & Hyperparameter Tuning

## Goal

Classify students into performance categories:

| Class     | Description               |
| --------- | ------------------------- |
| Fail      | Low performance           |
| Good      | Satisfactory performance  |
| Very Good | Above average performance |
| Excellent | Outstanding performance   |

---

# Classification Pipeline

## Data Preprocessing

* Missing value imputation
* Label encoding
* One-hot encoding
* Feature scaling using `StandardScaler`

## Classification Models

### Random Forest Classifier

A powerful ensemble tree-based classifier capable of capturing non-linear relationships between features.

### Logistic Regression

A linear probabilistic classifier used as a fast and interpretable baseline model.

### K-Nearest Neighbors (KNN)

An instance-based learning algorithm that classifies students based on similarity to neighboring samples.

---

# Classification Results

| Model               | Accuracy | Train Time | Test Time |
| ------------------- | -------- | ---------- | --------- |
| Random Forest       | 0.4702   | 7.58s      | 0.1643s   |
| Logistic Regression | 0.4368   | 5.27s      | 0.0053s   |
| KNN                 | 0.4418   | 0.03s      | 12.5669s  |

### Best Performing Model

Random Forest achieved the highest classification accuracy due to its ability to model complex feature interactions and non-linear educational patterns.

---

# Hyperparameter Tuning

The project includes systematic hyperparameter tuning experiments:

## Random Forest

* `n_estimators`
* `max_depth`

## Logistic Regression

* `C` regularization parameter

## KNN

* `n_neighbors`

The experiments demonstrated important machine learning concepts such as:

* Bias-variance tradeoff
* Underfitting vs overfitting
* Ensemble variance reduction
* Distance-based model sensitivity

---

# Key Findings & Insights

## Strong Predictors

* Historical academic performance
* VLE engagement activity
* Submission timing behavior
* Registration duration

## Educational Insights

* Students with higher VLE engagement achieved better scores.
* Late submissions negatively affected performance.
* Students who unregistered mid-course were highly associated with failure predictions.
* Historical performance was the strongest predictor of future success.

---

# Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Joblib
* Jupyter Notebook

---

# Machine Learning Concepts Applied

* Regression
* Classification
* Ensemble Learning
* Hyperparameter Tuning
* Feature Engineering
* Leakage Prevention
* Model Evaluation
* Exploratory Data Analysis
* Standardization & Scaling
* One-Hot Encoding

---

# Project Structure

```bash id="45a1vf"
project/
│
├── milestone1.py
├── milestone2.py
├── test_unseen.py
├── data/
├── models/
├── plots/
├── scaler.pkl
├── scaler_cls.pkl
├── fill_values.pkl
├── fill_values_cls.pkl
├── features.pkl
├── features_cls.pkl
├── label_encoder.pkl
└── README.md
```

---

# Future Improvements

* Deep Learning models
* XGBoost / LightGBM implementation
* Real-time student analytics dashboard
* Early warning system for at-risk students
* Model deployment using Flask or FastAPI
* Interactive educational reporting system

---

# Team Members

* Ahmed Medhat Mostafa Sayed Mohamed
* Alyeldeen Amr Aly Abdelfatah
* Youssef George Samuel
* Mohamed Ahmed Ismail

Faculty of Computer and Information Sciences (FCIS)
Ain Shams University
Department of Computer Science
Academic Year: 2025 / 2026

---

# Academic Context

Course: Machine Learning
Department: Computer Science
Project Phases:

* Milestone 1 — Regression & Preprocessing
* Milestone 2 — Classification & Hyperparameter Tuning

This project demonstrates a complete end-to-end educational data mining pipeline, including preprocessing, feature engineering, regression, classification, hyperparameter tuning, and performance evaluation using real-world learning analytics data.
