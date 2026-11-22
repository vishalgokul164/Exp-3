# EXPERIMENT 3: LOGISTIC REGRESSION FOR CLASSIFICATION

## AIM

To compute descriptive statistics and probabilities using NumPy and
Pandas.

## ALGORITHM

1.  *Start.*
2.  Load the dataset and extract features X and target y.
3.  Split the data into training and test sets (e.g., 80:20).
4.  Standardize features using StandardScaler.
5.  Train LogisticRegression on the training data.
    The logistic (sigmoid) function is:
    \[ \sigma{=tex}(z) = \frac{1}{1 + e^{-z}}{=tex} \]
    where
    \[ z = \beta{=tex}\_0 + \beta{=tex}\_1 x_1 + `\cdots `{=tex}+
    \beta{=tex}\_p x_p \]
6.  Predict labels on the test data.
7.  Evaluate performance using accuracy, confusion matrix, and
    classification report.
8.  Print the evaluation metrics.
9.  *Stop.*


## PROGRAM

import numpy as np  
import pandas as pd  
from sklearn.datasets import load_breast_cancer  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
print("--- Dataset Loading ---")  
cancer = load_breast_cancer()  
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)  
y = pd.Series(cancer.target)  
print(f"Dataset shape: {X.shape}")  
print("Target class counts:", np.unique(y, return_counts=True))  
X_train, X_test, y_train, y_test = train_test_split(  
X, y, test_size=0.2, random_state=42  
)  
print(f"\nTraining set size: {X_train.shape[0]} samples")  
print(f"Test set size: {X_test.shape[0]} samples")  
scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  
print("\n--- Model Training ---")  
model = LogisticRegression(random_state=42)  
model.fit(X_train_scaled, y_train)  
print("Logistic Regression model trained successfully.")  
y_pred = model.predict(X_test_scaled)  
print("\n--- Evaluation Metrics ---")  
accuracy = accuracy_score(y_test, y_pred)  
conf_matrix = confusion_matrix(y_test, y_pred)  
class_report = classification_report(y_test, y_pred)  
print(f"Accuracy Score: {accuracy:.4f}")  
print("\nConfusion Matrix:")  
print(conf_matrix)  
print("\nClassification Report:")  
print(class_report)  

## OUTPUT

<img width="433" height="432" alt="image" src="https://github.com/user-attachments/assets/47214671-4072-40c8-b10a-000b979c2196" />


## RESULT

The Logistic Regression model separates the standardized features cleanly, achieving an F1-score of 0.96 for malignant cases and 0.98 for benign cases.
