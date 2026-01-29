# Statistical Dependency Analysis Tool

## Overview
This project implements a primary dependency analysis tool for investment projects.
It is designed for exploratory analysis and early-stage decision support.

The tool identifies statistically significant dependencies between project features
and builds a proxy model for Net Present Value (NPV) estimation.

---

## Problem Statement
Input:
- Training and control datasets
- Categorical project features
- Target variable: NPV

Tasks:
1. Statistical dependency testing using Chi-square tests
2. Proxy NPV modeling using Linear Regression
3. Model evaluation on training and control samples
4. Visual correlation and prediction analysis

---

## Methods
- Chi-square hypothesis testing (α = 0.05)
- Manual implementation of Linear Regression (normal equation)
- Model validation using R² metric

---

## Tech Stack
Python, Pandas, NumPy, SciPy, Matplotlib, Seaborn, scikit-learn

---

## Repository Structure
