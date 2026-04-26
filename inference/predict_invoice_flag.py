import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "models/predict_flag_invoice.pkl"

# -----------------------------
# BASE FEATURES (FROM UI)
# -----------------------------
BASE_FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

# -----------------------------
# ENGINEERED FEATURES (USED IN TRAINING)
# -----------------------------
ENGINEERED_FEATURES = [
    "invoice_to_item_ratio",
    "quantity_ratio",
    "freight_ratio",
    "unit_price"
]

# FINAL FEATURE ORDER (MUST MATCH TRAINING EXACTLY)
FEATURES = BASE_FEATURES + ENGINEERED_FEATURES


def load_model(model_path: str = MODEL_PATH):
    with open(model_path, "rb") as f:
        return joblib.load(f)


# -----------------------------
# FEATURE ENGINEERING
# -----------------------------
def create_features(df):
    df = df.copy()

    # Avoid division errors
    df["invoice_to_item_ratio"] = df["invoice_dollars"] / (df["total_item_dollars"] + 1e-6)
    df["quantity_ratio"] = df["invoice_quantity"] / (df["total_item_quantity"] + 1e-6)
    df["freight_ratio"] = df["Freight"] / (df["invoice_dollars"] + 1e-6)
    df["unit_price"] = df["invoice_dollars"] / (df["invoice_quantity"] + 1e-6)

    return df


def predict_invoice_flag(input_data, threshold=0.7):
    """
    Predict invoice risk using ML probability scoring with threshold control.
    """

    model = load_model()

    # -----------------------------
    # Convert input → DataFrame
    # -----------------------------
    input_df = pd.DataFrame([input_data])

    # -----------------------------
    # Feature Engineering (IMPORTANT FIX)
    # -----------------------------
    input_df = create_features(input_df)

    # -----------------------------
    # Ensure correct feature order
    # -----------------------------
    input_df = input_df[FEATURES]

    # -----------------------------
    # Prediction probability
    # -----------------------------
    prob = model.predict_proba(input_df)[:, 1]

    # -----------------------------
    # Apply threshold
    # -----------------------------
    prediction = (prob >= threshold).astype(int)

    # -----------------------------
    # Output
    # -----------------------------
    input_df["Risk_Probability"] = prob
    input_df["Predicted_Flag"] = prediction

    return input_df


# -----------------------------
# LOCAL TEST
# -----------------------------
if __name__ == "__main__":

    sample_input = {
        "invoice_quantity": 120,
        "invoice_dollars": 5000.00,
        "Freight": 20.00,
        "total_item_quantity": 120,
        "total_item_dollars": 5000.00
    }

    result = predict_invoice_flag(sample_input)
    print(result)