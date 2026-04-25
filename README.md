# Student Productivity Prediction

A machine learning project designed to predict Student Productivity Scores by analyzing physiological data, sleep patterns, and academic performance metrics.

## Project Overview
This repository contains an end-to-end data science pipeline—from data preprocessing and feature engineering to model evaluation. The goal is to identify which factors (like REM sleep, heart rate, or GPA) most significantly impact a student's daily productivity.

## Dataset Breakdown
The model processes a variety of features extracted from student logs and wearable devices:
   * Demographics: Gender, University Year, Age.
   * Physiological Metrics: Heart Rate, Respiration Rate, Body Temperature.
   * Sleep Analysis: Duration, Quality, and Regularity.
   * Specific timings for Weekdays vs. Weekends.
   * Sleep Cycles (Light Movement vs. Rapid Eye Movement).
   * Academics: GPA and Study Hours Per Day.

## Tech Stack
 - Language: Python
 - Libraries:
      * pandas & numpy (Data Manipulation)
      * scikit-learn (Preprocessing & Traditional ML Models)
      * xgboost (Gradient Boosting)
      * matplotlib (Visualization)
      * joblib (Model & Feature Persistence)

## Getting Started
1. Clone the Repository
       git clone https://github.com/your-username/student-productivity-prediction.git
       cd student-productivity-prediction
2. Install Dependencies
       pip install pandas numpy matplotlib joblib scikit-learn xgboost openpyxl
3. Run the Training Notebook
   Open train_model.ipynb in Jupyter Lab or Notebook. The script performs:
      * Automated Feature Detection: Separates categorical and numerical data.
      * Preprocessing: Scales numerical data via StandardScaler and encodes categories via OneHotEncoder.
      * Cross-Model Benchmarking: Compares R² and MSE across 10+ algorithms including Random Forest, SVR, and Neural Networks (MLP).

## Evaluated Models
   - The project compares the performance of several architectures to find the best fit:
   - Linear Regressors: Linear, Ridge, Lasso.
   - Ensemble Methods: Random Forest, Extra Trees, Gradient Boosting, XGBoost.
   - Non-Linear: KNN, SVR, and Artificial Neural Networks (MLP).

## File Structure
 - train_model.ipynb: Core logic for training and evaluation.
 - categorical_cols.pkl / numeric_cols.pkl: Saved metadata for consistent inference.
 - student_productivity_predication_dataset.xlsx: Input data source.

## Model Performance & ResultsI evaluated 
I evaluated 11 different regression models. While several ensemble methods showed perfect training scores, the Multiple Linear Regression model achieved the best balance and the highest accuracy on unseen data (Test Set).Top Performers Comparison Model Test R² Test, RMSE Status 

| Model | Train $R^2$ | Test $R^2$ | Test RMSE | Status |
| :--- | :---: | :---: | :---: | :--- |
| **Multiple Linear Regression** | **0.8105** | **0.7804** | **0.4850** | **Best Fit** |
| Gradient Boosting | 0.9468 | 0.7611 | 0.5059 | Strong |
| Extra Trees | 1.0000 | 0.7608 | 0.5062 | Overfit |
| Random Forest | 0.9673 | 0.7442 | 0.5235 | Overfit |
| XGBoost | 0.9950 | 0.7314 | 0.5365 | Overfit |
| Lasso | 0.7573 | 0.7271 | 0.5408 | Underfit |
| Polynomial Ridge | 0.8483 | 0.7138 | 0.5537 | Decent |
| SVR | 0.9920 | 0.6757 | 0.5894 | High Var |
| KNN | 0.7577 | 0.5471 | 0.6966 | Poor |
| ANN | 0.8422 | 0.4758 | 0.7494 | Poor |
| Decision Tree | 1.0000 | 0.4330 | 0.7794 | High Overfit |

## Key Insights

  - The Winner: Multiple Linear Regression outperformed complex ensembles on the test set. This suggests the relationship between student habits (sleep, GPA, physiology) and productivity is relatively linear and the model isn't distracted by noise.
  - Overfitting Warning: Models like Decision Tree, Extra Trees, and XGBoost achieved a perfect $R^2$ of 1.0000 on training data but dropped significantly on test data. This indicates they "memorized" the training set rather than learning general patterns.
  - Deep Learning: The Artificial Neural Network (ANN) struggled with this specific dataset ($Test R^2: 0.3543$), likely due to the dataset size or requiring more hyperparameter tuning.
