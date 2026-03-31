# Heart Disease Risk Analyzer

## Live Demo

Try the app here: https://lhvpaw4nmtai5fsuytzaff.streamlit.app/

## Overview

A machine learning project that predicts heart disease risk from patient
measurements using the UCI Heart Disease Dataset.

## Dataset

- Source: UCI Heart Disease Dataset
- Combined 4 datasets: Cleveland, Hungary, Switzerland, Virginia
- 920 patients total
- 13 features per patient including age, cholesterol, blood pressure,
  and ECG measurements

## Features

- Data cleaning and imputation of missing values
- Exploratory data analysis and risk factor visualizations
- Two machine learning models compared
- Honest model evaluation using cross-validation

## Models

| Model               | Single Split Accuracy | Cross-Validation Accuracy |
| ------------------- | --------------------- | ------------------------- |
| Logistic Regression | 82%                   | -                         |
| Random Forest       | 84%                   | 77% (std: 0.08)           |

## Key Findings

- Most important predictors of heart disease: chest pain type,
  cholesterol, max heart rate, ST depression, and age
- Cross-validation revealed true accuracy is ~77%, showing the
  model needs more data to be reliable
- Dataset size (920 patients) is the main limiting factor

## What I Learned

- Data cleaning and handling missing medical data with imputation
- Difference between logistic regression and random forest models
- Hyperparameter tuning (n_estimators optimization)
- Why cross-validation gives a more honest accuracy than a
  single train/test split
- Feature importance analysis

## How to Run

1. Clone the repository
2. Install dependencies:
   pip install pandas numpy matplotlib scikit-learn
3. Add dataset files to the data/ folder
4. Run exploration.ipynb for visualizations
5. Run model.ipynb for predictions

## Project Structure

heart-disease-risk-analyzer/
├── data/
│ ├── processed.cleveland.data
│ ├── processed.hungarian.data
│ ├── processed.switzerland.data
│ └── processed.va.data
├── exploration.ipynb
├── model.ipynb
└── README.md

## Future Improvements

- Gather more patient data to improve reliability
- Try more advanced models
- Build a simple interface where you can input patient
  measurements and get a risk prediction

## Web Application

Run locally with:
pip install streamlit
streamlit run app.py
