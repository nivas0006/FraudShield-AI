# 🛡️ FraudShield AI

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-Ensemble-green)
![LightGBM](https://img.shields.io/badge/LightGBM-Gradient%20Boosting-brightgreen)
![SHAP](https://img.shields.io/badge/Explainable-AI-purple)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Completed-success)
![GitHub last commit](https://img.shields.io/github/last-commit/nivas0006/FraudShield-AI)
![GitHub repo size](https://img.shields.io/github/repo-size/nivas0006/FraudShield-AI)
![GitHub stars](https://img.shields.io/github/stars/nivas0006/FraudShield-AI?style=social)

> **An Explainable Machine Learning System for Financial Fraud Detection using Ensemble Learning, Model Comparison, and Explainable AI (SHAP).**


FraudShield AI is an end-to-end machine learning framework that detects fraudulent financial transactions using multiple publicly available datasets. The system integrates data preprocessing, feature engineering, model training, evaluation, explainability, and an interactive Streamlit dashboard into a unified pipeline.

---

# 📖 Table of Contents

- Overview
- Problem Statement
- Objectives
- Key Features
- Technologies Used
- System Architecture
- Workflow
- Project Structure
- Datasets
- Machine Learning Models
- Model Performance
- Explainable AI
- Dashboard
- Installation
- Usage
- Results
- Future Improvements
- License
- Author
- Acknowledgements

---

# 📌 Overview

FraudShield AI is designed to detect fraudulent financial transactions using supervised machine learning techniques. The project supports multiple fraud datasets, performs automated preprocessing, compares multiple machine learning models, selects the best-performing model, and explains predictions using SHAP (SHapley Additive Explanations).

Unlike traditional fraud detection systems that rely solely on rule-based methods, FraudShield AI combines modern ensemble learning algorithms with Explainable AI to improve both predictive performance and model transparency.

---

# ❗ Problem Statement

Financial fraud causes billions of dollars in losses annually. Detecting fraudulent transactions is difficult because fraudulent events represent only a tiny fraction of all transactions, resulting in highly imbalanced datasets.

Traditional rule-based systems struggle to adapt to evolving fraud patterns and often generate a high number of false positives.

FraudShield AI addresses these challenges by:

- Handling class imbalance using SMOTE
- Comparing multiple machine learning algorithms
- Selecting the optimal model automatically
- Providing interpretable predictions through Explainable AI (SHAP)

---

# 🎯 Objectives

- Detect fraudulent financial transactions with high accuracy.
- Build a reusable fraud detection framework.
- Support multiple public fraud datasets.
- Perform automated preprocessing and feature engineering.
- Compare multiple machine learning algorithms.
- Select the best-performing model automatically.
- Explain predictions using SHAP.
- Provide an interactive Streamlit dashboard for visualization.

---

# 🚀 Key Features

- Multi-dataset support
- Automated preprocessing pipeline
- Missing value handling
- Feature engineering
- Class balancing using SMOTE
- Five machine learning algorithms
- Automatic model comparison
- Model leaderboard
- Explainable AI using SHAP
- Fraud prediction
- Interactive Streamlit dashboard
- Dataset explorer
- Analytics dashboard
- Performance visualization
- Model evaluation reports

---

# 🛠 Technologies Used

## Programming Language

- Python 3

## Machine Learning

![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-orange?logo=scikitlearn)
![XGBoost](https://img.shields.io/badge/XGBoost-red)
![LightGBM](https://img.shields.io/badge/LightGBM-green)
![Imbalanced-Learn](https://img.shields.io/badge/Imbalanced--Learn-blueviolet)

## Explainable AI

- SHAP

## Data Analysis

- Pandas
- NumPy

## Visualization

- Matplotlib
- Plotly
- Streamlit

## Model Serialization

- Joblib

---

# 🏗 System Architecture

```
                    Financial Datasets
                            │
                            ▼
                    Data Loading Module
                            │
                            ▼
                  Data Preprocessing Module
                            │
                            ▼
                 Feature Engineering Module
                            │
                            ▼
                 Machine Learning Models
                            │
   ┌─────--─────┬─────--────┬─────--─────┬───-───────┬
   ▼            ▼           ▼            ▼           ▼
 Logistic   DecisionTree  RandomForest  XGBoost   LightGBM
 Regression
                            │
                            ▼
           Model Evaluation & Comparison
                            │
                            ▼
                 Best Model Selection
                            │
                            ▼
                 Explainable AI (SHAP)
                            │
                            ▼
                 Interactive Dashboard
```

---

# 🔄 Workflow

```
Datasets
     │
     ▼
Data Loading
     │
     ▼
    EDA
     │
     ▼
Preprocessing
     │
     ▼
Feature Engineering
     │
     ▼
   SMOTE
     │
     ▼
(only on)Train
     │
     ▼
    Test split
     │
     ▼
Model Training
     │
     ▼
Model Evaluation
     │
     ▼
Best Model Selection
     │
     ▼
Explainable AI
     │
     ▼
Dashboard
```

---

# 📂 Project Structure

```
FraudShield-AI
│
├── dashboard/
│   ├── app.py
│   └── pages/
│
├── datasets/
│
├── docs/
│
├── images/
│
├── models/
│
├── notebooks/
│
├── outputs/
│   ├── reports/
│   ├── model_leaderboard.csv
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── precision_recall.png
│
├── src/
│   ├── analysis/
│   ├── dashboard/
│   ├── data/
│   ├── evaluation/
│   ├── explainability/
│   ├── features/
│   ├── models/
│   └── utils/
│
├── tests/
│
├── README.md
├── requirements.txt
├── LICENSE
└── main.py
```

---

# 📊 Datasets

The project supports multiple fraud detection datasets.

### PaySim

A synthetic mobile money transaction dataset designed for fraud detection research.

### IEEE-CIS Fraud Detection

A real-world e-commerce fraud detection dataset containing transactional and identity information.

### Credit Card Fraud Detection

A highly imbalanced dataset containing European credit card transactions.

---

# 🤖 Machine Learning Models

The following models are implemented and evaluated.

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM

The best-performing model is automatically selected based on evaluation metrics.

---

# 📈 Model Performance

Five supervised machine learning models were trained and evaluated using the same preprocessing pipeline. Performance was measured using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Training Time.

| Rank | Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Training Time |
|------|----------------------|---------:|----------:|-------:|---------:|---------:|--------------:|
| 🥇 | Random Forest | **0.999807** | **0.999930** | **0.999683** | **0.999807** | **0.999999** | 65.52 s |
| 🥈 | XGBoost | 0.999754 | 0.999859 | 0.999648 | 0.999754 | 0.999964 | 1.48 s |
| 🥉 | LightGBM | 0.999727 | 0.999842 | 0.999613 | 0.999727 | 0.999960 | 1.27 s |
| 4 | Decision Tree | 0.999639 | 0.999648 | 0.999631 | 0.999639 | 0.999639 | 11.00 s |
| 5 | Logistic Regression | 0.998294 | 0.998996 | 0.997591 | 0.998293 | 0.999807 | 0.61 s |

## Best Performing Model

🏆 **Random Forest** achieved the highest overall performance among all evaluated models.

**Highlights**

- Accuracy: **99.98%**
- Precision: **99.99%**
- Recall: **99.97%**
- F1 Score: **99.98%**
- ROC-AUC: **99.9999%**

Although XGBoost and LightGBM achieved comparable predictive performance with significantly shorter training times, Random Forest demonstrated the strongest balance across all evaluation metrics and was therefore selected as the final deployment model.
# 🧠 Explainable AI

FraudShield AI incorporates SHAP (SHapley Additive Explanations) to improve model transparency.

The Explainable AI module provides:

- Feature importance
- SHAP summary plots
- Local prediction explanations
- Global model interpretation

This enables users to understand why the model classified a transaction as fraudulent.

---

# 📊 Dashboard

The Streamlit dashboard includes:

- Home
- Model Comparison
- Fraud Prediction
- Model Leaderboard
- Explainable AI
- Feature Importance
- Analytics
- Dataset Explorer
- About

---

# ⚙ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/FraudShield-AI.git

cd FraudShield-AI
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run dashboard/app.py
```

---

# ▶ Usage

1. Load a supported fraud dataset.
2. Perform preprocessing.
3. Train all machine learning models.
4. Compare model performance.
5. Select the best model.
6. Predict fraudulent transactions.
7. Explore SHAP explanations.
8. Analyze results using the dashboard.

---

# 📊 Experimental Results

The proposed FraudShield AI framework successfully:

- Trained and evaluated **five** supervised machine learning models.
- Automatically selected the highest-performing model.
- Achieved **99.98% Accuracy** on the evaluation dataset.
- Achieved **99.99% ROC-AUC**.
- Reduced the impact of class imbalance using **SMOTE**.
- Generated SHAP explanations for model transparency.
- Produced automated evaluation reports and a model leaderboard.
- Delivered an interactive Streamlit dashboard for fraud analysis.

---

# 🔮 Future Improvements

Possible future enhancements include:

- Deep Learning models
- Graph Neural Networks
- Real-time fraud detection
- REST API integration
- Docker deployment
- Kubernetes deployment
- Cloud deployment (AWS/Azure/GCP)
- MLOps pipeline
- Continuous model monitoring
- Drift detection
- Incremental learning
- Real-time transaction streaming

---

# 📜 License

This project is licensed under the MIT License.

---


#  Acknowledgements

This project utilizes publicly available datasets and open-source libraries.

Special thanks to:

- Scikit-Learn
- XGBoost
- LightGBM
- SHAP
- Streamlit
- Pandas
- NumPy
- Plotly
- Matplotlib
- IEEE-CIS Fraud Detection Dataset
- PaySim Dataset
- Credit Card Fraud Detection Dataset
