# Experiment 3: Logistic Regression for Classification

## Aim

To compute descriptive statistics and probabilities using NumPy and
Pandas.

## Algorithm

1.  **Start.**

2.  Load the dataset and extract features `X` and target `y`.

3.  Split the data into training and test sets (e.g., 80:20).

4.  Standardize features using `StandardScaler`.

5.  Train `LogisticRegression` on the training data.

    The logistic (sigmoid) function is:

    \[ `\sigma`{=tex}(z) = `\frac{1}{1 + e^{-z}}`{=tex} \]

    where

    \[ z = `\beta`{=tex}\_0 + `\beta`{=tex}\_1 x_1 + `\cdots `{=tex}+
    `\beta`{=tex}\_p x_p \]

6.  Predict labels on the test data.

7.  Evaluate performance using accuracy, confusion matrix, and
    classification report.

8.  Print the evaluation metrics.

9.  **Stop.**

 


## Program

# 1. Import required libraries   
import numpy as np  
import pandas as pd  

from sklearn.datasets import load_breast_cancer  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.linear_model import LogisticRegression  
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report  

# Display settings  
pd.set_option('display.max_columns', 8)  
pd.set_option('display.width', 120)  

# 2. Load the dataset and extract features (X) and target (y)
dataset = load_breast_cancer()

X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
y = pd.Series(dataset.target, name='target')

print("Dataset loaded successfully.")  
print("Feature shape:", X.shape)  
print("Target shape:", y.shape)  
print("\nTarget classes:")  
print(pd.Series(y).map({0: 'Malignant', 1: 'Benign'}).value_counts())  

# 3. Descriptive statistics using Pandas and NumPy
print("First five rows of the feature dataset:")
display(X.head())

print("\nDescriptive statistics:")
display(X.describe().round(2))

print("\nSelected NumPy statistics for mean radius:")  
radius = X['mean radius'].to_numpy()  
print("Mean:", round(np.mean(radius), 3))   
print("Median:", round(np.median(radius), 3))  
print("Standard deviation:", round(np.std(radius), 3))  
print("Minimum:", round(np.min(radius), 3))  
print("Maximum:", round(np.max(radius), 3))  

print("\nClass probabilities in the complete dataset:")  
class_probabilities = y.value_counts(normalize=True).sort_index()  
print("P(Malignant):", round(class_probabilities.loc[0], 4))  
print("P(Benign):", round(class_probabilities.loc[1], 4))  

# 4. Split the data into training and test sets (80:20)  
X_train, X_test, y_train, y_test = train_test_split(  
    X, y,  
    test_size=0.20,  
    random_state=42,  
    stratify=y  
)

print("Training feature shape:", X_train.shape)  
print("Testing feature shape :", X_test.shape)  
print("Training target shape:", y_train.shape)  
print("Testing target shape :", y_test.shape)  

# 5. Standardize the features
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test)  

print("Feature standardization completed.")   
print("Scaled training data shape:", X_train_scaled.shape)  
print("Scaled testing data shape :", X_test_scaled.shape)  

# 6. Train the Logistic Regression model  
model = LogisticRegression(max_iter=5000, random_state=42)  
model.fit(X_train_scaled, y_train)  

print("Logistic Regression model trained successfully.")  
print("Intercept (beta_0):", round(model.intercept_[0], 4))  
print("First five coefficients:", np.round(model.coef_[0][:5], 4))  

# 7. Understand the sigmoid function with a NumPy example  
def sigmoid(z):  
    return 1 / (1 + np.exp(-z))  
 
sample_z = np.array([-2, -1, 0, 1, 2])  
print("z values       :", sample_z)  
print("Sigmoid values :", np.round(sigmoid(sample_z), 4))  

# The model's predict_proba() method applies logistic probabilities  
# to estimate the probability of each class.  

# 8. Predict labels and probabilities on the test data  
y_pred = model.predict(X_test_scaled)  
y_prob = model.predict_proba(X_test_scaled)  

results = pd.DataFrame({  
    'Actual': y_test.to_numpy(),  
    'Predicted': y_pred,  
    'P(Malignant)': y_prob[:, 0],  
    'P(Benign)': y_prob[:, 1]  
}, index=y_test.index)  

print("First ten predictions with class probabilities:")  
display(results.head(10).round(4))  
****  

# 9. Evaluate model performance  
accuracy = accuracy_score(y_test, y_pred)  
cm = confusion_matrix(y_test, y_pred)  

print("Accuracy:", round(accuracy, 4))  
print("\nConfusion Matrix:")  
print(cm)   

print("\nClassification Report:")  
print(classification_report(  
    y_test, y_pred,  
    target_names=dataset.target_names,  
    digits=4  
))  

 





## output
Dataset loaded successfully.  
Feature shape: (569, 30)  
Target shape: (569,)  

Target classes:  
Benign       357  
Malignant    212  


## Result
