# student-performance-tracker
Student Performance Tracker built using Python and Machine Learning to analyze student data, predict academic performance, and visualize results through accuracy metrics and confusion matrices.
# Student Performance Tracker using Machine Learning

## Project Overview

This project predicts student performance using machine learning techniques. The system analyzes student-related factors such as gender, parental education, lunch type, test preparation course, and exam scores to determine whether a student is likely to pass or fail.

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn

## Machine Learning Algorithm

* Random Forest Classifier

## Features

* Data preprocessing
* Missing value handling
* Feature engineering
* Student performance prediction
* Accuracy evaluation
* Confusion matrix visualization
* Performance analytics

## Dataset

The project uses the Student Performance Dataset containing:

* Math Scores
* Reading Scores
* Writing Scores
* Gender
* Race/Ethnicity
* Parental Level of Education
* Lunch Type
* Test Preparation Course

## Project Workflow

1. Load dataset
2. Clean and preprocess data
3. Create performance labels (Pass/Fail)
4. Encode categorical variables
5. Train Random Forest model
6. Evaluate model performance
7. Visualize results

## Installation

pip install -r requirements.txt

## Run the Project

python student_performance.py

## Results

* Model Accuracy: Approximately 90%+ (varies by dataset split)
* Confusion Matrix Visualization
* Performance Prediction Reports

## Future Enhancements

* Streamlit Dashboard
* Student Ranking System
* Grade Prediction
* Attendance Analysis
* Performance Recommendations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================
# LOAD DATASET
# ==========================

df = pd.read_csv("StudentsPerformance.csv")

print("Dataset Shape:", df.shape)
print(df.head())

# ==========================
# CREATE TARGET COLUMN
# ==========================

# Pass if average marks >= 40

df["AverageScore"] = (
    df["math score"] +
    df["reading score"] +
    df["writing score"]
) / 3

df["Performance"] = np.where(
    df["AverageScore"] >= 40,
    "Pass",
    "Fail"
)

# ==========================
# FEATURES & TARGET
# ==========================

X = df.drop(
    columns=[
        "Performance",
        "AverageScore"
    ]
)

y = df["Performance"]

# Encode target
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# ==========================
# NUMERIC & CATEGORICAL
# ==========================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()

# ==========================
# PREPROCESSING
# ==========================

numeric_transformer = Pipeline(
    steps=[
        ("imputer",
         SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer",
         SimpleImputer(strategy="most_frequent")),
        ("onehot",
         OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num",
         numeric_transformer,
         numeric_features),

        ("cat",
         categorical_transformer,
         categorical_features)
    ]
)

# ==========================
# MODEL
# ==========================

model = Pipeline(
    steps=[
        ("preprocessor",
         preprocessor),

        ("classifier",
         RandomForestClassifier(
             n_estimators=300,
             random_state=42
         ))
    ]
)

# ==========================
# TRAIN TEST SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ==========================
# TRAIN
# ==========================

model.fit(X_train, y_train)

# ==========================
# PREDICT
# ==========================

y_pred = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:",
      round(accuracy * 100, 2),
      "%")

print("\nClassification Report:\n")
print(classification_report(
    y_test,
    y_pred
))

# ==========================
# ACCURACY GRAPH
# ==========================

plt.figure(figsize=(6,4))

plt.bar(
    ["Random Forest"],
    [accuracy * 100],
    color="green"
)

plt.ylabel("Accuracy (%)")
plt.title("Student Performance Prediction Accuracy")

plt.text(
    0,
    accuracy * 100 + 1,
    f"{accuracy*100:.2f}%"
)

plt.show()

# ==========================
# CONFUSION MATRIX
# ==========================

cm = confusion_matrix(
    y_test,
    y_pred
)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()

print("\nProject Completed Successfully!")
