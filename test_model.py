
import joblib
import pandas as pd
from pathlib import Path

# 1. Locate the saved model
MODEL_PATH = Path("artifacts/models/best_model.pkl")

# 2. Check whether the model file exists
if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model file not found: {MODEL_PATH.resolve()}"
    )

# 3. Load the saved model
model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")
print("Model type:", type(model).__name__)

# 4. Create a sample customer
# These columns must match the features used during training.
sample_customer = pd.DataFrame([{
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Male",
    "Age": 40,
    "Tenure": 5,
    "Balance": 75000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 80000.0,
    "Card Type": "DIAMOND",
    "Point Earned": 500,
    "Satisfaction Score": 3
}])

# 5. Generate a churn prediction
prediction = model.predict(sample_customer)

print("\nPredicted churn class:", prediction[0])

# 6. Generate churn probability, if supported
if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba(sample_customer)
    churn_probability = probabilities[0][1]

    print(
        f"Churn probability: "
        f"{churn_probability:.2%}"
    )
