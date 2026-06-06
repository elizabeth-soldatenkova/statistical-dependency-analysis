# Statistical Dependency Analysis Tool

## Overview

This project develops an end-to-end machine learning pipeline for forecasting Net Present Value (NPV) of oil & gas investment projects under geological uncertainty.

The pipeline combines statistical dependency analysis, feature engineering, ensemble learning, hyperparameter optimization, and model interpretability techniques to improve prediction accuracy and support investment decision-making.

## Problem Statement

### Input

* Training dataset
* Control dataset
* 10 categorical geological uncertainty factors
* Target variable: Net Present Value (NPV)

The categorical features represent probabilistic project scenarios using petroleum industry classifications such as P10, P50, P90, P10P50, and P50P90.

### Objectives

* Analyze dependencies between project factors
* Identify statistically significant relationships
* Build NPV forecasting models
* Compare classical and machine learning approaches
* Optimize model performance
* Interpret feature influence on project value

## Methodology

### Data Preprocessing

* Missing value handling
* Ordinal encoding of probabilistic categories
* Training/control dataset preparation

### Statistical Analysis

* Chi-square hypothesis testing (α = 0.05)
* Correlation analysis
* Feature dependency assessment

### Machine Learning Models

* Linear Regression (baseline model)
* Random Forest Regressor
* XGBoost Regressor
* LightGBM Regressor
* Multi-Layer Perceptron (MLP)

### Hyperparameter Optimization

* GridSearchCV
* RandomizedSearchCV
* Cross-validation

### Model Interpretation

* Feature Importance
* SHAP Values
* Bar and Beeswarm visualizations

## Results

* Evaluated 45 feature-pair relationships
* Identified 4 statistically significant dependencies
* Improved predictive performance from R² = 0.847 (Linear Regression) to R² = 0.966 (XGBoost / LightGBM)
* Compared five forecasting approaches
* Built an automated NPV forecasting workflow for oil & gas project screening
* Generated analytical visualizations supporting model interpretation

## Technologies

* Python
* Pandas
* NumPy
* SciPy
* Matplotlib
* Seaborn
* Scikit-learn
* XGBoost
* LightGBM
* SHAP

## Repository Structure

```text
├── data/
│   └── Task_1.xlsx
│
├── notebooks/
│   └── NPV_Forecasting_Pipeline.ipynb
│
├── figures/
│   ├── correlation_matrix.png
│   ├── model_comparison.png
│   └── shap_analysis.png
│
├── README.md
└── requirements.txt
```

## Key Outcome

The project demonstrates that ensemble gradient boosting models significantly outperform traditional linear approaches for NPV forecasting while maintaining model interpretability through Feature Importance and SHAP analysis.
