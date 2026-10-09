
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

st.title("🏦 Bank Customer Churn Prediction")
st.write(
    "Enter customer details below to estimate "
    "the probability of customer churn."
)

# --------------------------------------------------
# Load the trained model
# --------------------------------------------------
MODEL_PATH = (
    Path(__file__).resolve().parent
    / "artifacts"
    / "models"
    / "best_model.pkl"
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

try:
    model = load_model()
except Exception as error:
    st.error(f"Could not load the trained model: {error}")
    st.stop()

# --------------------------------------------------
# Customer input form
# --------------------------------------------------
with st.form("customer_form"):

    st.subheader("Customer Information")

    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650
        )

        geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"]
        )

        gender = st.selectbox(
            "Gender",
            ["Male", "Female"]
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=40
        )

        tenure = st.number_input(
            "Tenure (years)",
            min_value=0,
            max_value=10,
            value=5
        )

        balance = st.number_input(
            "Account Balance",
            min_value=0.0,
            value=75000.0,
            step=1000.0
        )

        num_products = st.selectbox(
            "Number of Products",
            [1, 2, 3, 4],
            index=1
        )

    with col2:
        has_card = st.selectbox(
            "Has Credit Card?",
            ["Yes", "No"]
        )

        active_member = st.selectbox(
            "Is Active Member?",
            ["Yes", "No"]
        )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=80000.0,
            step=1000.0
        )

        card_type = st.selectbox(
            "Card Type",
            ["DIAMOND", "GOLD", "PLATINUM", "SILVER"]
        )

        points_earned = st.number_input(
            "Points Earned",
            min_value=0,
            value=500,
            step=10
        )

        satisfaction_score = st.number_input(
            "Satisfaction Score",
            min_value=1,
            max_value=5,
            value=3
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        type="primary",
        use_container_width=True
    )

# --------------------------------------------------
# Prediction
# --------------------------------------------------
if submitted:

    customer = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": 1 if has_card == "Yes" else 0,
        "IsActiveMember": (
            1 if active_member == "Yes" else 0
        ),
        "EstimatedSalary": estimated_salary,
        "Card Type": card_type,
        "Point Earned": points_earned,
        "Satisfaction Score": satisfaction_score
    }])

    try:
        prediction = int(model.predict(customer)[0])

        st.divider()
        st.subheader("Prediction Results")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(customer)[0]
            churn_probability = float(probabilities[1])

            st.metric(
                "Churn Probability",
                f"{churn_probability:.2%}"
            )

            st.progress(churn_probability)

        if prediction == 1:
            st.error(
                "⚠️ Prediction: Customer is likely to churn."
            )
            st.write(
                "Consider reviewing customer engagement "
                "and retention options."
            )
        else:
            st.success(
                "✅ Prediction: Customer is unlikely to churn."
            )

        st.caption(
            "This is a model estimate, not a guarantee "
            "of future customer behaviour."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")
