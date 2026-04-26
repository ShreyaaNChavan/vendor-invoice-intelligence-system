# 📦 Vendor Invoice Intelligence System

## Live Demo

 **Deployed Application:**
[https://vendor-invoice-intelligence-system-tcc8.onrender.com](https://vendor-invoice-intelligence-system-tcc8.onrender.com)

---

##  Project Overview

The **Vendor Invoice Intelligence System** is an end-to-end Machine Learning application designed to:

* Predict Freight Cost for vendor invoices
* Detect risky or potentially fraudulent invoices
* Support financial decision-making using ML probability scoring
* Reduce manual invoice verification workload

The system is built using **Streamlit for UI**, **Scikit-learn for ML models**, and deployed on **Render Cloud Platform**.

---

## Problem Statement

Organizations process thousands of vendor invoices daily. Manual verification leads to:

* Human errors
* Delays in approval
* Financial leakage
* Inefficient auditing

This system automates invoice validation using machine learning.

---

## Objectives

* Predict freight cost based on invoice value
* Identify suspicious invoice patterns
* Provide risk score using ML probability
* Enable threshold-based decision-making
* Improve financial workflow efficiency

---

## Machine Learning Models Used

### Freight Cost Prediction

* Model: Linear Regression
* Input Features:

  * Invoice Dollars
* Output:

  * Predicted Freight Cost

---

### Invoice Risk Classification

* Model: Random Forest Classifier

* Input Features:

  * Invoice Quantity
  * Invoice Dollars
  * Freight Cost
  * Total Item Quantity
  * Total Item Dollars

* Output:

  * Risk Probability (0 to 1)
  * Fraud / Safe classification

---

##  Tech Stack

| Component       | Technology    |
| --------------- | ------------- |
| Frontend        | Streamlit     |
| Backend ML      | Scikit-learn  |
| Data Processing | Pandas, NumPy |
| Model Storage   | Joblib        |
| Deployment      | Render Cloud  |
| Language        | Python        |

---

## Project Structure

```text
ML PROJECT/
├── data/
│   └── inventory.db
├── freight_cost_predictions/
├── invoice_flagging/
├── inference/
│   ├── predict_freight.py
│   └── predict_invoice_flag.py
├── models/
│   ├── predict_flag_invoice.pkl
│   ├── predict_freight_model.pkl
│   └── scaler.pkl
├── notebooks/
├── app.py
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## ML Workflow

### 1. Data Preprocessing

* Missing value handling
* Feature scaling
* Feature selection

### 2. Model Training

* GridSearchCV for optimization
* Cross-validation
* F1-score optimization

### 3. Model Evaluation

* Accuracy ~ 88–91%
* Confusion Matrix analysis
* Precision/Recall balancing

### 4. Inference Pipeline

* Real-time input processing
* Probability scoring
* Risk classification

---

##  Risk Scoring Logic

The system outputs:

* **Risk Score (0–1)**
* Interpretation:

| Score Range | Meaning                  |
| ----------- | ------------------------ |
| 0.0 – 0.4   | Safe Invoice             |
| 0.4 – 0.7   | Medium Risk              |
| 0.7 – 1.0   | Manual Approval Required |

---

##  How to Run Locally

```bash
# Clone repo
git clone https://github.com/ShreyaaNChavan/vendor-invoice-intelligence-system

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py
```

---

## Deployment (Render)

---

##  Key Features

* Real-time invoice risk prediction
* ML-based fraud detection
* Interactive dashboard UI
* Probability-based decision system
* Cloud-deployed application

---

## Author

**Shreya Navnath Chavan**

AI & Data Science Student

---

##  Acknowledgement

This project demonstrates practical implementation of:

* Machine Learning in Finance
* Fraud Detection Systems
* End-to-end Deployment Pipeline

---
