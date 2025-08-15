import streamlit as st
import pandas as pd
import joblib

# Load model and encoders
model = joblib.load("employee_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("👨‍💼 Employee Attrition Predictor")

# Build input UI dynamically based on encoders
user_input = {}
for col, le in label_encoders.items():
    if col == "Attrition":
        continue  # Target variable, skip
    if hasattr(le, "classes_"):
        user_input[col] = st.selectbox(f"{col}", le.classes_)
    else:
        user_input[col] = st.number_input(f"{col}", step=1)

# Convert to DataFrame
input_df = pd.DataFrame([user_input])

# Encode categorical columns
for col, le in label_encoders.items():
    if col in input_df.columns and hasattr(le, "classes_"):
        input_df[col] = le.transform(input_df[col])

# Ensure order matches model training
# Ensure all required columns exist
for col in model.feature_names_in_:
    if col not in input_df.columns:
        input_df[col] = 0  # or a sensible default for that feature

# Reorder columns to match training

input_df = input_df[model.feature_names_in_]

# Predict
if st.button("Predict"):
    pred = model.predict(input_df)[0]
    pred_label = label_encoders["Attrition"].inverse_transform([pred])[0]
    st.success(f"Prediction: {pred_label}")
