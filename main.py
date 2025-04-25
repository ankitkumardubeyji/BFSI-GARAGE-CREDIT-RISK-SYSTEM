# main.py

import streamlit as st
import pandas as pd
import joblib

# Load the trained model and the feature column structure
model = joblib.load("credit_model.pkl")
feature_names = joblib.load("model_columns.pkl")

st.title("🏦 Credit Risk Predictor")

# Collect inputs
age = st.slider("Age", 18, 100)
job = st.selectbox("Job Type", [0, 1, 2, 3])
credit_amount = st.number_input("Credit Amount", min_value=0)
duration = st.slider("Duration (months)", 4, 72)
sex_male = st.selectbox("Sex", ["Male", "Female"]) == "Male"
housing_own = st.checkbox("Housing Own")
housing_rent = st.checkbox("Housing Rent")
saving_mod = st.checkbox("Saving Account: Moderate")
saving_qr = st.checkbox("Saving Account: Quite Rich")
saving_rich = st.checkbox("Saving Account: Rich")
saving_unknown = st.checkbox("Saving Account: Unknown")  # ✅ Added
checking_rich = st.checkbox("Checking Account: Rich")
checking_unknown = st.checkbox("Checking Account: Unknown")
checking_moderate = st.checkbox("Checking Account: Moderate")  # ✅ Added
purpose_car = st.checkbox("Purpose: Car")
purpose_domestic = st.checkbox("Purpose: Domestic Appliances")
purpose_edu = st.checkbox("Purpose: Education")
purpose_furn = st.checkbox("Purpose: Furniture/Equipment")
purpose_radio = st.checkbox("Purpose: Radio/TV")
purpose_repairs = st.checkbox("Purpose: Repairs")
purpose_vac = st.checkbox("Purpose: Vacation/Others")

# Create a DataFrame
input_dict = {
    'Age': age,
    'Job': job,
    'Credit amount': credit_amount,
    'Duration': duration,
    'Sex_male': sex_male,
    'Housing_own': housing_own,
    'Housing_rent': housing_rent,
    'Saving accounts_moderate': saving_mod,
    'Saving accounts_quite rich': saving_qr,
    'Saving accounts_rich': saving_rich,
    'Saving accounts_unknown': saving_unknown,
    'Checking account_rich': checking_rich,
    'Checking account_unknown': checking_unknown,
    'Checking account_moderate': checking_moderate,
    'Purpose_car': purpose_car,
    'Purpose_domestic appliances': purpose_domestic,
    'Purpose_education': purpose_edu,
    'Purpose_furniture/equipment': purpose_furn,
    'Purpose_radio/TV': purpose_radio,
    'Purpose_repairs': purpose_repairs,
    'Purpose_vacation/others': purpose_vac
}

input_df = pd.DataFrame([input_dict])

# Reindex to match model training columns
input_df = input_df.reindex(columns=feature_names, fill_value=0)

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    st.success("✔ Good Credit Risk" if prediction == 1 else "❌ Bad Credit Risk")
