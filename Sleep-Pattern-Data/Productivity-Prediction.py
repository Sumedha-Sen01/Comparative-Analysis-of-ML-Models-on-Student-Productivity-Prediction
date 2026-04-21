''' 
Linear Regression
Polynomial Regression (with Ridge)
Lasso Regression
Decision Tree
Random Forest
Extra Trees (Extremely Randomized Trees)
Gradient Boosting
XGBoost
K-Nearest Neighbors (KNN)
Support Vector Regression (SVR)
Artificial Neural Network (ANN / MLP)
'''

# STUDENT PRODUCTIVITY PREDICTION


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

# XGBoost
from xgboost import XGBRegressor



# 1. LOAD DATASET

print("LOADING DATASET")

df = pd.read_excel('Sleep-Pattern-Data/student_productivity_predication_dataset.xlsx')
print(df.head())
print(f"Dataset Shape: {df.shape}")
print("Dataset Loaded Successfully")


# 2. INPUT / OUTPUT SPLIT


X = df.drop(columns=["Student ID", "Productivity Score"])
y = df["Productivity Score"]


# 3. IDENTIFY COLUMN TYPES


categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# Save columns for Streamlit deployment
joblib.dump(categorical_cols, "categorical_cols.pkl")
joblib.dump(numeric_cols, "numeric_cols.pkl")

print("\nCategorical Columns:")
print(categorical_cols)

print("\nNumeric Columns:")
print(numeric_cols)



# 4. PREPROCESSOR


basic_preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_cols),
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
])

# Special preprocessor for Polynomial Ridge
poly_preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(degree=2, include_bias=False))
    ]), numeric_cols),

    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_cols)
])



# 5. TRAIN TEST SPLIT


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTrain-Test Split Completed")
print(f"Training Size : {X_train.shape}")
print(f"Testing Size  : {X_test.shape}")


# 6. DEFINE MODELS


models = {

    "Multiple Linear Regression": Pipeline([
        ("preprocessor", basic_preprocessor),
        ("regressor", LinearRegression())
    ]),

    "Polynomial Ridge": Pipeline([
        ("preprocessor", poly_preprocessor),
        ("regressor", Ridge(alpha=1.0))
    ]),

    "Lasso": Pipeline([
        ("preprocessor", basic_preprocessor),
        ("regressor", Lasso(alpha=0.1))
    ]),

    "Decision Tree": Pipeline([
        ("preprocessor", basic_preprocessor),
        ("regressor", DecisionTreeRegressor(random_state=42))
    ]),

    "Random Forest": Pipeline([
        ("preprocessor", basic_preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ))
    ]),

    "Extra Trees": Pipeline([
        ("preprocessor", basic_preprocessor),
        ("regressor", ExtraTreesRegressor(
            n_estimators=200,
            random_state=42
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("preprocessor", basic_preprocessor),
        ("regressor", GradientBoostingRegressor(
            n_estimators=200,
            random_state=42
        ))
    ]),

    "XGBoost": Pipeline([
        ("preprocessor", basic_preprocessor),
        ("regressor", XGBRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42,
            verbosity=0
        ))
    ]),

    "KNN": Pipeline([
        ("preprocessor", basic_preprocessor),
        ("regressor", KNeighborsRegressor(
            n_neighbors=5
        ))
    ]),

    "SVR": Pipeline([
        ("preprocessor", basic_preprocessor),
        ("regressor", SVR(
            kernel="rbf",
            C=100,
            gamma="scale",
            epsilon=0.1
        ))
    ]),

    "ANN": Pipeline([
        ("preprocessor", basic_preprocessor),
        ("regressor", MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            max_iter=1500,
            early_stopping=True,
            random_state=42
        ))
    ])
}



# 7. MODEL TRAINING + EVALUATION


print("MODEL COMPARISON RESULTS")

results = []

for name, model in models.items():
    print(f"\nTraining Model: {name}")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    results.append({
        "Model": name,
        "Train MSE": round(train_mse, 4),
        "Test MSE": round(test_mse, 4),
        "Train RMSE": round(train_rmse, 4),
        "Test RMSE": round(test_rmse, 4),
        "Train R²": round(train_r2, 4),
        "Test R²": round(test_r2, 4)
    })

    print(f"Train R² : {train_r2:.4f}")
    print(f"Test R²  : {test_r2:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")


# 8. SAVE RESULTS


results_df = pd.DataFrame(results)
results_df = results_df.sort_values("Test R²", ascending=False)

results_df.to_csv("model_results.csv", index=False)

print("\nResults saved as: model_results.csv")



# 9. SAVE ALL MODELS


print("\nSaving Trained Models...")

for name, model in models.items():
    filename = name.lower().replace(" ", "_") + ".pkl"
    joblib.dump(model, filename)
    print(f"Saved: {filename}")



# 10. VISUALIZATION FUNCTION


def comparison_graph(metric_train, metric_test, title, ylabel, filename):
    x = np.arange(len(results_df["Model"]))
    width = 0.35

    plt.figure(figsize=(14, 8))

    plt.bar(
        x - width / 2,
        results_df[metric_train],
        width,
        label=metric_train,
        alpha=0.8
    )

    plt.bar(
        x + width / 2,
        results_df[metric_test],
        width,
        label=metric_test,
        alpha=0.8
    )

    plt.xticks(
        x,
        results_df["Model"],
        rotation=45,
        ha="right",
        fontsize=10
    )

    plt.xlabel("Models", fontsize=12, fontweight="bold")
    plt.ylabel(ylabel, fontsize=12, fontweight="bold")
    plt.title(title, fontsize=14, fontweight="bold")

    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Saved Graph: {filename}")



# 11. GENERATE COMPARISON GRAPHS


print("\nComparison Graphs")

comparison_graph(
    "Train R²",
    "Test R²",
    "R² Score Comparison Across All Models",
    "R² Score",
    "r2_comparison.png"
)

comparison_graph(
    "Train MSE",
    "Test MSE",
    "MSE Comparison Across All Models",
    "Mean Squared Error",
    "mse_comparison.png"
)

comparison_graph(
    "Train RMSE",
    "Test RMSE",
    "RMSE Comparison Across All Models",
    "Root Mean Squared Error",
    "rmse_comparison.png"
)

print("\nPROJECT COMPLETED SUCCESSFULLY")
