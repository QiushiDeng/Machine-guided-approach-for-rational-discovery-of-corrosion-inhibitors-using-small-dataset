#!/usr/bin/env python
# coding: utf-8

# ## Import blocks

# In[8]:


import numpy as np
import pandas as pd
import seaborn as sns
import time
import torch
import random
from sklearn.svm import SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.inspection import permutation_importance


# ## GBR_uncluster

# In[ ]:


feature_columns = ['latent_1', 'latent_2', 'latent_3', 'latent_4', 'latent_5', 
                  'latent_6', 'latent_7', 'latent_8', 'latent_9', 'latent_10',
                  'conc(mM)', 'HOMO (eV)','LUMO (eV)','Gap (eV)','elevated HOMO',
                   'alleviated LUMO','Num_atom_FG','vertical ionisation potential (IP)',
                  'Electron affinity (A)', 'absolute chemical hardness (g)','electron charge transfer (delta N)',
                  'change in energy','electrophilicity index (w)','dipole (p) unit Debye','isotropic polarisability (a)']
                  #   ,'Num_atom_FG','vertical ionisation potential (IP)',
                  #  'Electron affinity (A)','electron chemical potential (u)',
                  #  'absolute chemical hardness (g)','electron charge transfer (delta N)',
                  #  'change in energy','electrophilicity index (w)', 'dipole (p) unit Debye'
                  # ]
X = cluster_subset[feature_cols].values
y = cluster_subset['IE'].values.reshape(-1, 1)
# Check if the columns exist in the dataframe
missing_features = [col for col in feature_columns if col not in df.columns]
if missing_features:
    print(f"Warning: The following specified features don't exist in the dataset: {missing_features}")
    print("Available columns are:", df.columns.tolist())
    exit()

if target_column not in df.columns:
    print(f"Error: Target column '{target_column}' not found in the dataset")
    print("Available columns are:", df.columns.tolist())
    exit()

# Prepare the data
X = df[feature_columns]
y = df[target_column]

param_grid = {
    'n_estimators': [10, 50, 100],
    'learning_rate': [0.01],
    'subsample': [0.6, 0.8, 1.0],
    'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'random_state': [42]  # Fixed for reproducibility
}

# Create the GradientBoostingRegressor model
gbr = GradientBoostingRegressor()

# Create the GridSearchCV
print("\nStarting GridSearchCV. This may take some time...")
start_time = time.time()
grid_search = GridSearchCV(
    estimator=gbr,
    param_grid=param_grid,
    cv=3,  # 3-fold cross-validation
    scoring='neg_mean_squared_error',
    verbose=2,
    n_jobs=-1  # Use all available processors
)

# Fit the GridSearchCV
grid_search.fit(X, y)
end_time = time.time()
print(f"GridSearchCV completed in {end_time - start_time:.2f} seconds")

# Print the best parameters and best score
print("\nBest Parameters:")
print(grid_search.best_params_)
print(f"Best Cross-Validation Score (negative MSE): {grid_search.best_score_:.4f}")

# Get the best model
best_gbr = grid_search.best_estimator_

# Make predictions on test set
y_pred = best_gbr.predict(X)

# Evaluate the model
mse = mean_squared_error(y,y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print("\nModel Evaluation on Test Set:")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Feature importance
feature_importance = pd.DataFrame(
    {'Feature': feature_columns, 
     'Importance': best_gbr.feature_importances_}
).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)


# ## Extreme random forest

# In[ ]:


# Select features and target
feature_cols = ['latent_1', 'latent_2', 'latent_3', 'latent_4', 'latent_5', 
                  'latent_6', 'latent_7', 'latent_8', 'latent_9', 'latent_10',
                  'conc(mM)', 'HOMO (eV)','LUMO (eV)','Gap (eV)','elevated HOMO',
                   'alleviated LUMO','Num_atom_FG','vertical ionisation potential (IP)',
                  'Electron affinity (A)', 'absolute chemical hardness (g)','electron charge transfer (delta N)',
                  'change in energy','electrophilicity index (w)','dipole (p) unit Debye','isotropic polarisability (a)']
X = cluster_subset[feature_cols].values
y = cluster_subset['IE'].values.reshape(-1, 1)

# Standardize features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Standardize target
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Define GridSearchCV parameters
param_grid = {
    'n_estimators': [50, 100],           
    'max_depth': [None, 10],             
    'min_samples_split': [2,3,4,5],         
    'min_samples_leaf': [1, 2,4,8],          
    'max_features': [0.5, 0.7]           
}

print("\n=== Starting Grid Search ===")
print("Parameter grid:")
for param, values in param_grid.items():
    print(f"{param}: {values}")

base_model = ExtraTreesRegressor(random_state=42, n_jobs=2)
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,                    
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=1                 
)

# Fit the grid search
try:
    print("\nRunning grid search... This may take some time.")
    grid_search.fit(X_scaled, y_scaled.ravel())

    # Get best parameters and results
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert back to positive RMSE

    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best RMSE during grid search: {best_score:.4f}")

except KeyboardInterrupt:
    print("\nGrid search was interrupted. Using default parameters instead.")
    # Default parameters if grid search is interrupted
    best_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 0.5
    }
    print("\nUsing default parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")



kf = KFold(n_splits=3, shuffle=True, random_state=42)


cv_model = ExtraTreesRegressor(random_state=42, n_jobs=2, **best_params)


cv_preds_scaled = cross_val_predict(cv_model, X_scaled, y_scaled.ravel(), cv=kf)
cv_preds_scaled = cv_preds_scaled.reshape(-1, 1)

cv_rmse = np.sqrt(mean_squared_error(y_scaled, cv_preds_scaled))
cv_mae = mean_absolute_error(y_scaled, cv_preds_scaled)
cv_r2 = r2_score(y_scaled, cv_preds_scaled)
cv_preds_original = scaler_y.inverse_transform(cv_preds_scaled)
y_original = scaler_y.inverse_transform(y_scaled)

cv_rmse_original = np.sqrt(mean_squared_error(y_original, cv_preds_original))
cv_mae_original = mean_absolute_error(y_original, cv_preds_original)
cv_r2_original = r2_score(y_original, cv_preds_original)

fold_metrics = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
    X_train_fold, X_test_fold = X_scaled[train_idx], X_scaled[test_idx]
    y_train_fold, y_test_fold = y_scaled[train_idx], y_scaled[test_idx]
    fold_model = ExtraTreesRegressor(random_state=42, n_jobs=2, **best_params)
    fold_model.fit(X_train_fold, y_train_fold.ravel())
    fold_preds = fold_model.predict(X_test_fold).reshape(-1, 1)
    fold_rmse = np.sqrt(mean_squared_error(y_test_fold, fold_preds))
    fold_r2 = r2_score(y_test_fold, fold_preds)

    print(f"Fold {fold}: RMSE = {fold_rmse:.4f}, R² = {fold_r2:.4f}")
    fold_metrics.append((fold_rmse, fold_r2))

# Calculate average and standard deviation of fold metrics
fold_rmses, fold_r2s = zip(*fold_metrics)
print(f"\nFold RMSE - Mean: {np.mean(fold_rmses):.4f}, Std: {np.std(fold_rmses):.4f}")
print(f"Fold R² - Mean: {np.mean(fold_r2s):.4f}, Std: {np.std(fold_r2s):.4f}")

final_model = ExtraTreesRegressor(random_state=42, n_jobs=2, **best_params)
final_model.fit(X_scaled, y_scaled.ravel())

# Get predictions on the full dataset
full_preds_scaled = final_model.predict(X_scaled).reshape(-1, 1)
full_rmse = np.sqrt(mean_squared_error(y_scaled, full_preds_scaled))
full_mae = mean_absolute_error(y_scaled, full_preds_scaled)
full_r2 = r2_score(y_scaled, full_preds_scaled)

print(f"RMSE: {full_rmse:.4f}, MAE: {full_mae:.4f}, R²: {full_r2:.4f}")

full_preds_original = scaler_y.inverse_transform(full_preds_scaled)
full_rmse_original = np.sqrt(mean_squared_error(y_original, full_preds_original))
full_mae_original = mean_absolute_error(y_original, full_preds_original)
full_r2_original = r2_score(y_original, full_preds_original)


# ## Supporting vector

# In[ ]:


feature_cols = ['latent_1', 'latent_2', 'latent_3', 'latent_4', 'latent_5', 
                  'latent_6', 'latent_7', 'latent_8', 'latent_9', 'latent_10',
                  'conc(mM)', 'HOMO (eV)','LUMO (eV)','Gap (eV)','elevated HOMO',
                   'alleviated LUMO','Num_atom_FG','vertical ionisation potential (IP)',
                  'Electron affinity (A)', 'absolute chemical hardness (g)','electron charge transfer (delta N)',
                  'change in energy','electrophilicity index (w)','dipole (p) unit Debye','isotropic polarisability (a)']

# Standardize features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Standardize target
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Define GridSearchCV parameters for SVR
param_grid = {
    'kernel': ['rbf'],           
    'C': [0.1, 1.0, 10.0, 100],             
    'epsilon': [0.1, 0.2],       
    'gamma': ['scale', 'auto']          
}

for param, values in param_grid.items():
    print(f"{param}: {values}")

# Create the base model (SVR)
base_model = SVR()

# Setup GridSearchCV with 3-fold CV
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,                    
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=1                 
)

try:
    print("\nRunning grid search... This may take some time.")
    grid_search.fit(X_scaled, y_scaled.ravel())

    # Get best parameters and results
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert back to positive RMSE

    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best RMSE during grid search: {best_score:.4f}")

kf = KFold(n_splits=3, shuffle=True, random_state=42)

cv_model = SVR(**best_params)


cv_preds_scaled = cross_val_predict(cv_model, X_scaled, y_scaled.ravel(), cv=kf)
cv_preds_scaled = cv_preds_scaled.reshape(-1, 1)
cv_rmse = np.sqrt(mean_squared_error(y_scaled, cv_preds_scaled))
cv_mae = mean_absolute_error(y_scaled, cv_preds_scaled)
cv_r2 = r2_score(y_scaled, cv_preds_scaled)

cv_preds_original = scaler_y.inverse_transform(cv_preds_scaled)
y_original = scaler_y.inverse_transform(y_scaled)

cv_rmse_original = np.sqrt(mean_squared_error(y_original, cv_preds_original))
cv_mae_original = mean_absolute_error(y_original, cv_preds_original)
cv_r2_original = r2_score(y_original, cv_preds_original)

fold_metrics = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
    X_train_fold, X_test_fold = X_scaled[train_idx], X_scaled[test_idx]
    y_train_fold, y_test_fold = y_scaled[train_idx], y_scaled[test_idx]

    # Train on this fold
    fold_model = SVR(**best_params)
    fold_model.fit(X_train_fold, y_train_fold.ravel())

    # Predict on test set
    fold_preds = fold_model.predict(X_test_fold).reshape(-1, 1)

    # Calculate metrics
    fold_rmse = np.sqrt(mean_squared_error(y_test_fold, fold_preds))
    fold_r2 = r2_score(y_test_fold, fold_preds)

    print(f"Fold {fold}: RMSE = {fold_rmse:.4f}, R² = {fold_r2:.4f}")
    fold_metrics.append((fold_rmse, fold_r2))

# Calculate average and standard deviation of fold metrics
fold_rmses, fold_r2s = zip(*fold_metrics)
print(f"\nFold RMSE - Mean: {np.mean(fold_rmses):.4f}, Std: {np.std(fold_rmses):.4f}")
print(f"Fold R² - Mean: {np.mean(fold_r2s):.4f}, Std: {np.std(fold_r2s):.4f}")

# Get predictions on the full dataset
full_preds_scaled = final_model.predict(X_scaled).reshape(-1, 1)
full_rmse = np.sqrt(mean_squared_error(y_scaled, full_preds_scaled))
full_mae = mean_absolute_error(y_scaled, full_preds_scaled)
full_r2 = r2_score(y_scaled, full_preds_scaled)

# Convert to original scale
full_preds_original = scaler_y.inverse_transform(full_preds_scaled)
full_rmse_original = np.sqrt(mean_squared_error(y_original, full_preds_original))
full_mae_original = mean_absolute_error(y_original, full_preds_original)
full_r2_original = r2_score(y_original, full_preds_original)


# ## Ridger regressmor

# In[ ]:


feature_cols = ['latent_1', 'latent_2', 'latent_3', 'latent_4', 'latent_5', 
                  'latent_6', 'latent_7', 'latent_8', 'latent_9', 'latent_10',
                  'conc(mM)', 'HOMO (eV)','LUMO (eV)','Gap (eV)','elevated HOMO',
                   'alleviated LUMO','Num_atom_FG','vertical ionisation potential (IP)',
                  'Electron affinity (A)', 'absolute chemical hardness (g)','electron charge transfer (delta N)',
                  'change in energy','electrophilicity index (w)','dipole (p) unit Debye','isotropic polarisability (a)']


# Standardize features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0],         # Regularization strength
    'kernel': ['linear'],     # Kernel types
    'gamma': [0.001, 0.01, 0.1, 1],    # Kernel coefficient for rbf and poly
    'degree': [2, 3]                         # Polynomial degree (only for poly kernel)
}
print("\n=== Starting Grid Search ===")
print("Parameter grid:")
for param, values in param_grid.items():
    print(f"{param}: {values}")

# Create the base model (Kernel Ridge Regression)
base_model = KernelRidge()

# Setup GridSearchCV with 3-fold CV
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,                    
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=1                 
)

# Fit the grid search
try:
    print("\nRunning grid search... This may take some time.")
    grid_search.fit(X_scaled, y_scaled.ravel())

    # Get best parameters and results
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert back to positive RMSE

    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best RMSE during grid search: {best_score:.4f}")
except Exception as e:
    print("Grid search failed with an error:", e) 
# ======= Perform 3-fold Cross-Validation with Best Parameters =======
print("\n=== Performing 3-Fold Cross-Validation with Best Parameters ===")

# Define the 3-fold cross-validation
kf = KFold(n_splits=3, shuffle=True, random_state=42)

# Create model with best parameters
cv_model = KernelRidge(**best_params)

# Use cross_val_predict to get predictions for each fold
cv_preds_scaled = cross_val_predict(cv_model, X_scaled, y_scaled.ravel(), cv=kf)
cv_preds_scaled = cv_preds_scaled.reshape(-1, 1)
cv_rmse = np.sqrt(mean_squared_error(y_scaled, cv_preds_scaled))
cv_mae = mean_absolute_error(y_scaled, cv_preds_scaled)
cv_r2 = r2_score(y_scaled, cv_preds_scaled)

# Convert to original scale for cross-validation results
cv_preds_original = scaler_y.inverse_transform(cv_preds_scaled)
y_original = scaler_y.inverse_transform(y_scaled)

cv_rmse_original = np.sqrt(mean_squared_error(y_original, cv_preds_original))
cv_mae_original = mean_absolute_error(y_original, cv_preds_original)
cv_r2_original = r2_score(y_original, cv_preds_original)

fold_metrics = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
    X_train_fold, X_test_fold = X_scaled[train_idx], X_scaled[test_idx]
    y_train_fold, y_test_fold = y_scaled[train_idx], y_scaled[test_idx]

    # Train on this fold
    fold_model = KernelRidge(**best_params)
    fold_model.fit(X_train_fold, y_train_fold.ravel())

    # Predict on test set
    fold_preds = fold_model.predict(X_test_fold).reshape(-1, 1)

    # Calculate metrics
    fold_rmse = np.sqrt(mean_squared_error(y_test_fold, fold_preds))
    fold_r2 = r2_score(y_test_fold, fold_preds)

    print(f"Fold {fold}: RMSE = {fold_rmse:.4f}, R² = {fold_r2:.4f}")
    fold_metrics.append((fold_rmse, fold_r2))

# Calculate average and standard deviation of fold metrics
fold_rmses, fold_r2s = zip(*fold_metrics)
print(f"\nFold RMSE - Mean: {np.mean(fold_rmses):.4f}, Std: {np.std(fold_rmses):.4f}")
print(f"Fold R² - Mean: {np.mean(fold_r2s):.4f}, Std: {np.std(fold_r2s):.4f}")

final_model = KernelRidge(**best_params)
final_model.fit(X_scaled, y_scaled.ravel())

# Get predictions on the full dataset
full_preds_scaled = final_model.predict(X_scaled).reshape(-1, 1)
full_rmse = np.sqrt(mean_squared_error(y_scaled, full_preds_scaled))
full_mae = mean_absolute_error(y_scaled, full_preds_scaled)
full_r2 = r2_score(y_scaled, full_preds_scaled)
# Convert to original scale
full_preds_original = scaler_y.inverse_transform(full_preds_scaled)

full_rmse_original = np.sqrt(mean_squared_error(y_original, full_preds_original))
full_mae_original = mean_absolute_error(y_original, full_preds_original)
full_r2_original = r2_score(y_original, full_preds_original)


# ## Random forest

# In[ ]:


feature_cols = ['latent_1', 'latent_2', 'latent_3', 'latent_4', 'latent_5', 
                  'latent_6', 'latent_7', 'latent_8', 'latent_9', 'latent_10',
                  'conc(mM)', 'HOMO (eV)','LUMO (eV)','Gap (eV)','elevated HOMO',
                   'alleviated LUMO','Num_atom_FG','vertical ionisation potential (IP)',
                  'Electron affinity (A)', 'absolute chemical hardness (g)','electron charge transfer (delta N)',
                  'change in energy','electrophilicity index (w)','dipole (p) unit Debye','isotropic polarisability (a)']

# Standardize features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)

# Define GridSearchCV parameters for Random Forest
param_grid = {
    'n_estimators': [50, 100, 200],            # Number of trees in the forest
    'max_depth': [None, 5, 10],               # Maximum depth of trees
    'min_samples_split': [2, 5],           # Minimum samples required to split a node
    'min_samples_leaf': [1, 2, 4],             # Minimum samples required at a leaf node
    'max_features': ['sqrt', 'log2', None]     # Number of features to consider for best split
}

print("\n=== Starting Grid Search ===")
print("Parameter grid:")
for param, values in param_grid.items():
    print(f"{param}: {values}")

# Create the base model (Random Forest)
base_model = RandomForestRegressor(random_state=42, n_jobs=2)

# Setup GridSearchCV with 3-fold CV
grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    cv=3,                    
    scoring='neg_root_mean_squared_error',
    verbose=1,
    n_jobs=1                 
)

# Fit the grid search
try:
    print("\nRunning grid search... This may take some time.")
    grid_search.fit(X_scaled, y_scaled.ravel())

    # Get best parameters and results
    best_params = grid_search.best_params_
    best_score = -grid_search.best_score_  # Convert back to positive RMSE

    print("\nBest Hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best RMSE during grid search: {best_score:.4f}")

except KeyboardInterrupt:
    print("\nGrid search was interrupted. Using default parameters instead.")
    # Default parameters if grid search is interrupted
    best_params = {
        'n_estimator¶s': 100,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1,
        'max_features': 'sqrt'
    }
    print("\nUsing default parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")

kf = KFold(n_splits=3, shuffle=True, random_state=42)

cv_model = RandomForestRegressor(random_state=42, n_jobs=2, **best_params)
cv_preds_scaled = cross_val_predict(cv_model, X_scaled, y_scaled.ravel(), cv=kf)
cv_preds_scaled = cv_preds_scaled.reshape(-1, 1)
cv_rmse = np.sqrt(mean_squared_error(y_scaled, cv_preds_scaled))
cv_mae = mean_absolute_error(y_scaled, cv_preds_scaled)
cv_r2 = r2_score(y_scaled, cv_preds_scaled)

print("\n=== Cross-Validation Performance (Standardized Scale) ===")
print(f"RMSE: {cv_rmse:.4f}, MAE: {cv_mae:.4f}, R²: {cv_r2:.4f}")

# Convert to original scale for cross-validation results
cv_preds_original = scaler_y.inverse_transform(cv_preds_scaled)
y_original = scaler_y.inverse_transform(y_scaled)

cv_rmse_original = np.sqrt(mean_squared_error(y_original, cv_preds_original))
cv_mae_original = mean_absolute_error(y_original, cv_preds_original)
cv_r2_original = r2_score(y_original, cv_preds_original)

fold_metrics = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled), 1):
    X_train_fold, X_test_fold = X_scaled[train_idx], X_scaled[test_idx]
    y_train_fold, y_test_fold = y_scaled[train_idx], y_scaled[test_idx]

    # Train on this fold
    fold_model = RandomForestRegressor(random_state=42, n_jobs=2, **best_params)
    fold_model.fit(X_train_fold, y_train_fold.ravel())

    # Predict on test set
    fold_preds = fold_model.predict(X_test_fold).reshape(-1, 1)

    # Calculate metrics
    fold_rmse = np.sqrt(mean_squared_error(y_test_fold, fold_preds))
    fold_r2 = r2_score(y_test_fold, fold_preds)

    print(f"Fold {fold}: RMSE = {fold_rmse:.4f}, R² = {fold_r2:.4f}")
    fold_metrics.append((fold_rmse, fold_r2))

# Calculate average and standard deviation of fold metrics
fold_rmses, fold_r2s = zip(*fold_metrics)
print(f"\nFold RMSE - Mean: {np.mean(fold_rmses):.4f}, Std: {np.std(fold_rmses):.4f}")
print(f"Fold R² - Mean: {np.mean(fold_r2s):.4f}, Std: {np.std(fold_r2s):.4f}")

# Get predictions on the full dataset
full_preds_scaled = final_model.predict(X_scaled).reshape(-1, 1)
full_rmse = np.sqrt(mean_squared_error(y_scaled, full_preds_scaled))
full_mae = mean_absolute_error(y_scaled, full_preds_scaled)
full_r2 = r2_score(y_scaled, full_preds_scaled)
full_preds_original = scaler_y.inverse_transform(full_preds_scaled)
full_rmse_original = np.sqrt(mean_squared_error(y_original, full_preds_original))
full_mae_original = mean_absolute_error(y_original, full_preds_original)
full_r2_original = r2_score(y_original, full_preds_original)

