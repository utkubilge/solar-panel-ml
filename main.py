import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso,
    ElasticNet,
    PassiveAggressiveRegressor,
    HuberRegressor,
    TheilSenRegressor,
    BayesianRidge,
    SGDRegressor,
    PoissonRegressor,
)
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    StackingRegressor,
    VotingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from sklearn.preprocessing import StandardScaler
import time
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor


# 1. Load the data for both plants
weather_df1 = pd.read_csv("./data/Plant_1_Weather_Sensor_Data.csv")
generation_df1 = pd.read_csv("./data/Plant_1_Generation_Data.csv")
weather_df2 = pd.read_csv("./data/Plant_2_Weather_Sensor_Data.csv")
generation_df2 = pd.read_csv("./data/Plant_2_Generation_Data.csv")

# Convert DATE_TIME to datetime objects
weather_df1["DATE_TIME"] = pd.to_datetime(weather_df1["DATE_TIME"])
generation_df1["DATE_TIME"] = pd.to_datetime(generation_df1["DATE_TIME"], dayfirst=True)
weather_df2["DATE_TIME"] = pd.to_datetime(weather_df2["DATE_TIME"])
generation_df2["DATE_TIME"] = pd.to_datetime(generation_df2["DATE_TIME"])

# Merge the datasets on 'DATE_TIME' and 'PLANT_ID'
merged_df1 = pd.merge(generation_df1, weather_df1, on=["DATE_TIME"])
merged_df2 = pd.merge(generation_df2, weather_df2, on=["DATE_TIME"])

# Combine both plants' data
merged_df = pd.concat([merged_df1, merged_df2], ignore_index=True)

# Remove rows where AC_POWER is zero
merged_df = merged_df[merged_df["AC_POWER"] != 0]

# Select features and target
features = merged_df[["AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]
target = merged_df["AC_POWER"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Scale features for models that need it
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define regression models (added more ensemble and linear models)
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(),
    "Lasso Regression": Lasso(),
    "ElasticNet": ElasticNet(),
    "PassiveAggressive": PassiveAggressiveRegressor(max_iter=1000, random_state=42),
    "HuberRegressor": HuberRegressor(),
    "TheilSen": TheilSenRegressor(random_state=42),
    "BayesianRidge": BayesianRidge(),
    "SGDRegressor": SGDRegressor(max_iter=1000, tol=1e-3, random_state=42),
    "PoissonRegressor": PoissonRegressor(max_iter=1000),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
    "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=100, random_state=42),
    "AdaBoost": AdaBoostRegressor(n_estimators=100, random_state=42),
    "Bagging": BaggingRegressor(n_estimators=100, random_state=42),
    "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42),
    "K-Nearest Neighbors": KNeighborsRegressor(n_neighbors=5),
    "Voting Regressor": VotingRegressor([
        ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
        ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
        ('dt', DecisionTreeRegressor(random_state=42))
    ]),
    "Stacking Regressor": StackingRegressor(
        estimators=[
            ('rf', RandomForestRegressor(n_estimators=50, random_state=42)),
            ('gb', GradientBoostingRegressor(n_estimators=50, random_state=42)),
            ('dt', DecisionTreeRegressor(random_state=42))
        ],
        final_estimator=LinearRegression()
    ),
    "LightGBM": LGBMRegressor(n_estimators=100, random_state=42),
    "CatBoost": CatBoostRegressor(n_estimators=100, random_state=42, verbose=0),
    "XGBoost": XGBRegressor(n_estimators=100, random_state=42),
}

# Evaluate models and collect metrics
results_list = []
for name, model in models.items():
    start_time = time.time()
    # Use scaled data for models that benefit from it
    if name in [
        "Linear Regression",
        "Ridge Regression",
        "Lasso Regression",
        "ElasticNet",
        "K-Nearest Neighbors",
    ]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elapsed_time = time.time() - start_time
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    results_list.append({
        "Model": name,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "MAE": mae,
        "Time (s)": elapsed_time
    })

# Sort by R2 descending
results_list = sorted(results_list, key=lambda x: x["R2"], reverse=True)

print("Model Performance Comparison (sorted by R2):")
for res in results_list:
    print(
        f"{res['Model']}: R2={res['R2']:.4f}, MSE={res['MSE']:.2f}, RMSE={res['RMSE']:.2f}, MAE={res['MAE']:.2f}, Time={res['Time (s)']:.2f}s"
    )

# Optional: Show a few predictions from the best model (highest R2)
best_model_name = results_list[0]["Model"]
best_model = models[best_model_name]
if best_model_name in [
    "Linear Regression",
    "Ridge Regression",
    "Lasso Regression",
    "ElasticNet",
    "K-Nearest Neighbors",
]:
    best_model.fit(X_train_scaled, y_train)
    y_pred_best = best_model.predict(X_test_scaled)
else:
    best_model.fit(X_train, y_train)
    y_pred_best = best_model.predict(X_test)
results = pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred_best})
print(f"\nSample predictions (best model - {best_model_name}):")
print(results.head())
