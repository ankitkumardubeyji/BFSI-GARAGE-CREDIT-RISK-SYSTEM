# 💳 Credit Risk Prediction Using Random Forest Classifier

A Machine Learning project that classifies loan applicants as **"Good Credit Risk"** or **"Bad Credit Risk"** based on demographic and financial attributes. The project is powered by a **Random Forest Classifier** and deployed via **Streamlit** for real-time prediction.

---

## 🎯 Objective

To develop an accurate credit scoring model that aids in financial decision-making by classifying loan applicants into risk categories based on input features.

---

## 📊 Dataset Description

The dataset contains **22 columns**:
- **Numerical Features**: `Age`, `Job`, `Credit amount`, `Duration`
- **Binary/Categorical Encoded Features**:
  - **Gender**: `Sex_male`
  - **Housing**: `Housing_own`, `Housing_rent`
  - **Saving Accounts**: `Saving accounts_moderate`, `Saving accounts_quite rich`, `Saving accounts_rich`
  - **Checking Accounts**: `Checking account_rich`, `Checking account_unknown`
  - **Purpose**: Multiple one-hot encoded columns like `Purpose_car`, `Purpose_education`, etc.
- **Target Variable**: `Risk`  
  - `0 = Bad Credit`  
  - `1 = Good Credit`

---

## ⚙️ Methodology

### 1. Data Preprocessing
- One-hot encoded categorical features (already preprocessed).
- Feature matrix `X` and label vector `y` were separated.
- Data split into **80% training** and **20% testing** using `train_test_split`.

### 2. Model Selection: `RandomForestClassifier`
- **Why Random Forest?**
  - Reduces overfitting through ensemble learning.
  - Handles both numerical and categorical variables well.
  - Robust, scalable, and performs well out of the box.

### 3. Model Evaluation
- Predictions were generated on the test set (`X_test`).
- Evaluation metrics used:
  - **Confusion Matrix**
  - **Classification Report** (Precision, Recall, F1-Score, Accuracy)

# Evaluation Matrix
  Confusion Matrix:
[[ 70   0]
 [  0 130]]

Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        70
           1       1.00      1.00      1.00       130

    accuracy                           1.00       200
   macro avg       1.00      1.00      1.00       200
weighted avg       1.00      1.00      1.00       200


### 4. Model Saving
```python
import joblib
joblib.dump(model, "credit_model.pkl")
