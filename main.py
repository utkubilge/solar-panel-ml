import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Load data
gen = pd.read_csv('./data/Plant_2_Generation_Data.csv')
weather = pd.read_csv('./data/Plant_2_Weather_Sensor_Data.csv')

# Merge on 'DATE_TIME' and 'PLANT_ID'
data = pd.merge(gen, weather, on=['DATE_TIME'])

# Convert DATE_TIME to datetime and extract features
data['DATE_TIME'] = pd.to_datetime(data['DATE_TIME'])
data['HOUR'] = data['DATE_TIME'].dt.hour
data['DAY'] = data['DATE_TIME'].dt.day
data['MONTH'] = data['DATE_TIME'].dt.month
data['WEEKDAY'] = data['DATE_TIME'].dt.weekday

# Remove night-time rows (when AC_POWER is zero)
data = data[data['AC_POWER'] > 0]

# Feature interaction
data['IRRADIATION_X_MODULE_TEMP'] = data['IRRADIATION'] * data['MODULE_TEMPERATURE']

# Additional feature engineering
data['TEMP_DIFF'] = data['MODULE_TEMPERATURE'] - data['AMBIENT_TEMPERATURE']
data['HOUR_SIN'] = np.sin(2 * np.pi * data['HOUR'] / 24)
data['HOUR_COS'] = np.cos(2 * np.pi * data['HOUR'] / 24)

# Select features and target
features = [
    'AMBIENT_TEMPERATURE',
    'MODULE_TEMPERATURE',
    'IRRADIATION',
    'HOUR',
    'DAY',
    'MONTH',
    'WEEKDAY',
    'IRRADIATION_X_MODULE_TEMP',
    'TEMP_DIFF',
    'HOUR_SIN',
    'HOUR_COS'
]
X = data[features]
y = data['AC_POWER']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train LightGBM with more estimators and tuned parameters
lgbm = LGBMRegressor(n_estimators=300, learning_rate=0.05, max_depth=7, random_state=42)
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'R2 Score: {r2:.4f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')

# Plot Actual vs Predicted
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual AC_POWER')
plt.ylabel('Predicted AC_POWER')
plt.title('LightGBM: Actual vs Predicted AC_POWER')
plt.show()

# Residual plot for error analysis
residuals = y_test - y_pred
plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted AC_POWER')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

# Train Random Forest with tuned parameters
rf = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Metrics
r2 = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f'R2 Score: {r2:.4f}')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')

# Plot Actual vs Predicted
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred, alpha=0.3)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual AC_POWER')
plt.ylabel('Predicted AC_POWER')
plt.title('Random Forest: Actual vs Predicted AC_POWER')
plt.show()

# Residual plot for error analysis
residuals = y_test - y_pred
plt.figure(figsize=(8,5))
plt.scatter(y_pred, residuals, alpha=0.3)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Predicted AC_POWER')
plt.ylabel('Residuals')
plt.title('Random Forest: Residual Plot')
plt.show()