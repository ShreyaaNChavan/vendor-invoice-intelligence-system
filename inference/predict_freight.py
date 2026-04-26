import joblib
import pandas as pd

MODEL_PATH = "models/predict_freight_model.pkl"

# ✔ Feature used in training
FEATURES = ["Dollars"]

def load_model(model_path: str = MODEL_PATH):
    """
    Load trained freight cost prediction model.
    """
    with open(model_path, "rb") as f:
        model = joblib.load(f)
    return model


def predict_freight_cost(input_data):
    """
    Predict freight cost for new vendor invoices.

    Parameters
    ----------
    input_data : dict (single record)

    Returns
    -------
    pd.DataFrame with predicted freight cost
    """
    model = load_model()

    # ✔ FIX 1: wrap dict in list to avoid pandas scalar error
    input_df = pd.DataFrame([input_data])

    # ✔ FIX 2: ensure correct feature order
    input_df = input_df[FEATURES]

    # ✔ prediction
    input_df["Predicted_Freight"] = model.predict(input_df).round()

    return input_df


if __name__ == "__main__":

    # ✔ Example inference run (single input)
    sample_data = {
        "Dollars": 18500.00   # single scalar (correct for Streamlit-style input)
    }

    prediction = predict_freight_cost(sample_data)
    print(prediction)