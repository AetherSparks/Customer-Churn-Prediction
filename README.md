# Customer Churn Prediction

This project aims to predict customer churn using various machine learning models, including logistic regression, random forests, and boosted trees. We use feature engineering, visualization, and model evaluation techniques to analyze and interpret customer churn data, specifically from the Telco Customer dataset.

## Table of Contents
Project Overview
Dataset
Installation
Data Preprocessing
Exploratory Data Analysis
Model Training and Evaluation
Hyperparameter Tuning
Feature Importance and SHAP Analysis
Usage
Results
Conclusion

## Project Overview
This project uses the Telco Customer dataset to build a predictive model for customer churn. We analyze customer characteristics and services, preprocess the data, and apply various classification models to identify patterns associated with churn. Models are evaluated for accuracy, precision, recall, F1-score, and ROC-AUC.

## Dataset
The dataset used is Telco Customer Churn. Key features include customer demographics, service subscriptions, and account information. The dataset is processed to make TotalCharges numeric, drop irrelevant fields (e.g., customerID), and encode categorical variables.

## Installation
To run this project, install the necessary packages by executing:

pip install shap pandas numpy seaborn matplotlib plotly scikit-learn xgboost

## Data Preprocessing
Steps involved in data preprocessing include:

Checking for missing values and handling them by removing or imputing data.
Converting categorical features using one-hot encoding.
Scaling numerical features using StandardScaler to normalize data for model training.
## Exploratory Data Analysis
We analyze and visualize key features in the dataset, such as:

Correlation Heatmap: Using Plotly to understand feature relationships.
Gender and Churn Distribution: Visualizing churn rates by gender and other demographics.
Churn by Service Subscriptions: Checking the churn tendency among different service groups.
## Model Training and Evaluation
We apply multiple models for churn prediction, including:

Logistic Regression
Random Forest
XGBoost
Support Vector Machine (SVM)
Decision Tree and others.
Performance metrics for each model are calculated to determine the best model.

## Hyperparameter Tuning
For models like RandomForest and XGBoost, Grid Search with Cross-Validation is used to fine-tune parameters.

## Feature Importance and SHAP Analysis
Using SHAP values, we analyze feature importance to understand the drivers of customer churn and plot SHAP summary plots for further interpretation.

## Usage
To use this model:

Load Dataset: Upload your dataset to WA_Fn-UseC_-Telco-Customer-Churn.csv.
Run Preprocessing Steps: Prepare the data by following the preprocessing steps in the notebook.
Train Models and Evaluate: Choose a model and evaluate its performance.
Save the Model: Save the best model with joblib for later use.

import joblib
joblib.dump(model, 'logistic_regression_churn_model.pkl')
## Results
The best-performing model was the Random Forest classifier, achieving a high accuracy and recall. The logistic regression model also performed well, providing interpretability through SHAP.

## Conclusion
This project demonstrates an approach to churn prediction using multiple classification models and SHAP interpretation, providing insight into customer retention and churn strategies.
