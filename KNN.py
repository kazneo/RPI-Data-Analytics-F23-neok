import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


df = pd.read_csv('/data/preproessed_collected_dataset_flood.csv')


df['Year'] = df['DATE'].apply(lambda x: x // 10000)
df = df.groupby(['LONGITUDE', 'LATITUDE', 'Year']).apply(lambda x: x.sample(frac=0.01))


dummies = pd.get_dummies(df, columns=['FLD_ZONE', 'CATEGORY', 'STATE'])



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