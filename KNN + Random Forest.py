# Imoprt Packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Read Data
df = pd.read_csv('/data/preproessed_collected_dataset_flood.csv')


df['Year'] = df['DATE'].apply(lambda x: x // 10000)
df = df.groupby(['LONGITUDE', 'LATITUDE', 'Year']).apply(lambda x: x.sample(frac=0.01))


dummies = pd.get_dummies(df, columns=['FLD_ZONE', 'CATEGORY', 'STATE'])

# Select Featuers
feature_names=['Elevation', 'Wind_f', 'Evap', 'Tair_f', 'Qair_f', 'Psurf_f',
               'Streamflow', 'SoilMoist100_200cm', 'SoilTemp100_200cm', 'LC_Type2',
               'FLD_ELEV', 'Qsb', 'CFLD_RISKS',
               'RFLD_RISKS', 'HRCN_RISKS', 'Rainf_f_MA30',
               'FLD_ZONE_A', 'FLD_ZONE_AE', 'FLD_ZONE_AH', 'FLD_ZONE_AO',
               'FLD_ZONE_Nan', 'FLD_ZONE_VE', 'FLD_ZONE_X',
               'FLD_ZONE_X PROTECTED BY LEVEE']

X = dummies[feature_names]

y = dummies['FloodedFrac']

# 80% Traning 10% Testing  10% Valitation Split 
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=10)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=10)



k_values = range(1, 21)

# Lists to store MSE values for different K values
mse_values_val = []  # MSE values for validation set
mse_values_test = []  # MSE values for test set

for k in k_values:
    print(k)
    # Initialize the KNN regressor with the current K value
    reg = KNeighborsRegressor(n_neighbors=k)
    
    # Fit the model on the training set
    reg.fit(X_train, y_train)
    
    # Predict on the validation set
    y_val_pred = reg.predict(X_val)
    
    # Calculate mean squared error on validation set and store it
    mse_val = mean_squared_error(y_val, y_val_pred)
    mse_values_val.append(mse_val)
    
    # Predict on the test set
    y_test_pred = reg.predict(X_test)
    
    # Calculate mean squared error on test set and store it
    mse_test = mean_squared_error(y_test, y_test_pred)
    mse_values_test.append(mse_test)

# Plotting the elbow curves
plt.figure(figsize=(8, 6))

# Plot for validation set MSE
plt.plot(k_values, mse_values_val, marker='o', linestyle='-', color='b', label='Validation Set')

# Plot for test set MSE
plt.plot(k_values, mse_values_test, marker='o', linestyle='-', color='r', label='Test Set')

plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Mean Squared Error (MSE)')
plt.xticks(k_values)
plt.grid(True)
plt.legend()
plt.show()

params = {
    'n_neighbors': 4,
    'weights': 'distance',
    'p': 1,  # For Minkowski distance (1 for Manhattan, 2 for Euclidean)
    'algorithm': 'auto',
}

reg = KNeighborsRegressor(**params)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


reg.fit(X_train_scaled, y_train)


knn_train_pred = reg.predict(X_train_scaled)
knn_val_pred = reg.predict(X_val_scaled)
knn_test_pred = reg.predict(X_test_scaled)


knn_val_mse = mean_squared_error(y_val, knn_val_pred)
knn_test_mse = mean_squared_error(y_test, knn_test_pred)

knn_val_rmse = np.sqrt(knn_val_mse)
knn_test_rmse = np.sqrt(knn_test_mse)


print("Validation MSE:", knn_val_mse)
print("Test MSE:", knn_test_mse)

print("Validation RMSE:", knn_val_rmse)
print("Test RMSE:", knn_test_rmse)



plt.scatter(y_test, knn_test_pred, alpha=0.6, color='blue', edgecolors='w')

plt.xlabel("Actual y values")
plt.ylabel("Predicted y values")
plt.title('KNN - Actual vs Predicted')

ax = plt.gca()
ax.set_aspect('equal')

# Set the limits for better visualization
lims = [
    np.min([y_test.min(), knn_test_pred.min()]),
    np.max([y_test.max(), knn_test_pred.max()])
]
plt.xlim(lims)
plt.ylim(lims)

# Plot the 45-degree line
plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

plt.grid(True)
plt.show()
plt.clf()  # Clear the current figure window


# Random Forest Parameters
params = {
    "n_estimators": 500,
    "max_features": 8,
    "max_depth": 15,
    "min_samples_split": 2,
    "warm_start":True,
    "oob_score":True,
    "random_state": 42,
    "verbose" : 1,
    "n_jobs" : -1,
}

reg = RandomForestRegressor(**params)

reg.fit(X_train, y_train)


rf_train_pred = reg.predict(X_train_scaled)
rf_val_pred = reg.predict(X_val_scaled)
rf_test_pred = reg.predict(X_test_scaled)


rf_val_mse = mean_squared_error(y_val, rf_val_pred)
rf_test_mse = mean_squared_error(y_test, rf_test_pred)

rf_val_rmse = np.sqrt(rf_val_mse)
rf_test_rmse = np.sqrt(rf_test_mse)

print("Validation MSE:", rf_val_mse)
print("Test MSE:", rf_test_mse)

print("Validation RMSE:", rf_val_rmse)
print("Test RMSE:", rf_test_rmse)



plt.scatter(y_test, rf_test_pred, alpha=0.6, color='blue', edgecolors='w')

plt.xlabel("Actual y values")
plt.ylabel("Predicted y values")
plt.title('RF - Actual vs Predicted')

ax = plt.gca()
ax.set_aspect('equal')

# Set the limits for better visualization
lims = [
    np.min([y_test.min(), rf_test_pred.min()]),
    np.max([y_test.max(), rf_test_pred.max()])
]
plt.xlim(lims)
plt.ylim(lims)

# Plot the 45-degree line
plt.plot(lims, lims, 'k-', alpha=0.75, zorder=0)

plt.grid(True)
plt.show()
plt.clf()  # Clear the current figure window



# obtain feature importance
feature_importance = reg.feature_importances_

sorted_idx = np.argsort(feature_importance)[::-1]
pos = np.arange(sorted_idx.shape[0])

# Set the figure size to have more space between bars
plt.figure(figsize=(10, 8)) 

# Plot feature importances with increased spacing between bars
plt.barh(pos, feature_importance[sorted_idx], align="center")

plt.yticks(pos, np.array(feature_names)[sorted_idx])
plt.title("Feature Importance (MDI)")
plt.xlabel("Mean decrease in impurity")
plt.tight_layout()
plt.show()

