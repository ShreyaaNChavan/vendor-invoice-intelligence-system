from data_preprocessing import (
    load_invoice_data,
    apply_labels,
    split_data
)

from model_evaluation import train_random_forest, evaluate_classifier
import joblib

FEATURES = [
    "invoice_quantity",
    "invoice_dollars",
    "Freight",
    "total_item_quantity",
    "total_item_dollars"
]

TARGET = "flag_invoice"


def main():

    # -------------------------
    # Load data
    # -------------------------
    df = load_invoice_data()
    df = apply_labels(df)

    # -------------------------
    # (IMPORTANT) FEATURE ENGINEERING GOES HERE
    # -------------------------
    df["invoice_to_item_ratio"] = df["invoice_dollars"] / df["total_item_dollars"]
    df["quantity_ratio"] = df["invoice_quantity"] / df["total_item_quantity"]
    df["freight_ratio"] = df["Freight"] / df["invoice_dollars"]
    df["unit_price"] = df["invoice_dollars"] / df["invoice_quantity"]

    # update features list
    FEATURES_EXTENDED = FEATURES + [
        "invoice_to_item_ratio",
        "quantity_ratio",
        "freight_ratio",
        "unit_price"
    ]

    # -------------------------
    # Split data
    # -------------------------
    X_train, X_test, y_train, y_test = split_data(df, FEATURES_EXTENDED, TARGET)

    # -------------------------
    # Train model (NO SCALING)
    # -------------------------
    grid_search = train_random_forest(X_train, y_train)

    # -------------------------
    # Evaluate
    # -------------------------
    evaluate_classifier(
        grid_search.best_estimator_,
        X_test,
        y_test,
        "Random Forest Classifier"
    )

    # -------------------------
    # Save model
    # -------------------------
    joblib.dump(grid_search.best_estimator_, 'models/predict_flag_invoice.pkl')


if __name__ == "__main__":
    main()