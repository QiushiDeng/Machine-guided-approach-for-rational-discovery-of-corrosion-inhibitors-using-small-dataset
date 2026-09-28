#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import warnings
import numpy as np
import torch
import tensorflow as tf
from sklearn.model_selection import KFold, GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential, save_model, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from scikeras.wrappers import KerasRegressor
from tensorflow.keras.callbacks import EarlyStopping
from scipy.spatial.distance import euclidean

centroids = np.load('filtered_centroids.npy')
print(f"Loaded centroids shape: {centroids.shape}")

submodels = {
    'ERF': joblib.load('sub_models/ERF_model_cluster_12.0.joblib'),
    'GBR': joblib.load('sub_models/GBR_model_cluster_14.0.joblib'),
    'KRR_1': joblib.load('sub_models/KRR_model_cluster_2.0.joblib'), 
    'KRR_2': joblib.load('sub_models/KRR_model_cluster_10.0.joblib'),
    'RF': joblib.load('sub_models/RF_model_cluster_11.0.joblib'),
    'SVR_1': joblib.load('sub_models/SVR_model_cluster_0.0.joblib'),
    'SVR_2': joblib.load('sub_models/SVR_model_cluster_15.0.joblib'),
    'XGB_1': joblib.load('sub_models/XGB_model_cluster_3.0.joblib'),
    'XGB_2': joblib.load('sub_models/XGB_model_cluster_9.0.joblib')
}

cluster_ids = {}
for model_name in submodels.keys():
    # Extract from filename pattern
    if 'cluster_' in model_name:
        try:
            cluster_ids[model_name] = float(model_name.split('cluster_')[1].split('.joblib')[0])
        except:
            pass
    else:
        # Try to find cluster in the model name
        for part in model_name.split('_'):
            try:
                cluster_ids[model_name] = float(part)
                break
            except:
                pass

    # If still not found, use the index as fallback
    if model_name not in cluster_ids:
        cluster_ids[model_name] = list(submodels.keys()).index(model_name)

def predict_with_submodel_flexible(X_data, model_obj, model_name, debug=True):
    """Handle both dictionary and direct model formats"""

    if debug:
        print(f"\nProcessing {model_name}:")

    # Check if it's a dictionary or direct model
    if isinstance(model_obj, dict):
        # Dictionary format (ERF, KRR_1, etc.)
        model = model_obj['model']
        x_scaler = model_obj.get('scaler_X')
        y_scaler = model_obj.get('scaler_y')
        feature_cols = model_obj.get('feature_cols')

        if debug:
            print(f"  Format: Dictionary")
            print(f"  Has scaler_X: {x_scaler is not None}")
            print(f"  Has scaler_y: {y_scaler is not None}")
            print(f"  Has feature_cols: {feature_cols is not None}")
    else:
        # Direct model format (GBR)
        model = model_obj
        x_scaler = None  # No scaling available
        y_scaler = None  # No scaling available

        # Get feature columns from model
        if hasattr(model, 'feature_names_in_'):
            feature_cols = model.feature_names_in_
        else:
            feature_cols = None

        if debug:
            print(f"  Format: Direct model")
            print(f"  Has scaler_X: False")
            print(f"  Has scaler_y: False")
            print(f"  Has feature_cols: {feature_cols is not None}")

    # Feature selection
    if feature_cols is not None:
        missing_cols = [col for col in feature_cols if col not in X_data.columns]
        if missing_cols and debug:
            print(f"  Warning: Missing {len(missing_cols)} columns")

        # Create temporary DataFrame with zeros for missing columns
        X_temp = X_data.copy()
        for col in missing_cols:
            X_temp[col] = 0

        X_selected = X_temp[feature_cols]
    else:
        X_selected = X_data

    # Apply X scaling if available
    if x_scaler is not None:
        try:
            X_scaled = x_scaler.transform(X_selected.values)
            if debug:
                print(f"  X scaled shape: {X_scaled.shape}")
                print(f"  X scaled range: [{np.min(X_scaled):.4f}, {np.max(X_scaled):.4f}]")
        except Exception as e:
            if debug:
                print(f"  Error in X scaling: {e}")
            X_scaled = X_selected.values
    else:
        X_scaled = X_selected.values
        if debug:
            print(f"  No X scaling available - using raw data")

    # Get raw predictions
    try:
        raw_preds = model.predict(X_scaled)

        if debug:
            print(f"  Raw predictions shape: {raw_preds.shape}")
            print(f"  Raw predictions range: [{np.min(raw_preds):.4f}, {np.max(raw_preds):.4f}]")
    except Exception as e:
        if debug:
            print(f"  Error in prediction: {e}")
        return np.zeros(len(X_data))

    # Apply inverse Y scaling if available
    if y_scaler is not None:
        try:
            if len(raw_preds.shape) == 1:
                reshaped_preds = raw_preds.reshape(-1, 1)
            else:
                reshaped_preds = raw_preds

            inversed_preds = y_scaler.inverse_transform(reshaped_preds)

            if len(inversed_preds.shape) > 1 and inversed_preds.shape[1] == 1:
                final_preds = inversed_preds.flatten()
            else:
                final_preds = inversed_preds   
            if debug:
                print(f"  Inverse transformed predictions shape: {final_preds.shape}")
                print(f"  Inverse transformed range: [{np.min(final_preds):.4f}, {np.max(final_preds):.4f}]")

        except Exception as e:
            if debug:
                print(f"  Error in Y inverse scaling: {e}")
                print(f"  Using raw predictions instead")
            final_preds = raw_preds
    else:
        if debug:
            print(f"  No Y scaling available - using raw predictions")
        final_preds = raw_preds

    return final_preds

# Update get_meta_model_features to use the flexible function
def get_meta_model_features_flexible(X_data, debug=True):
    """Get meta-model features using flexible submodel prediction"""
    n_samples = X_data.shape[0]
    n_models = len(submodels)
    model_names = list(submodels.keys())

    if debug:
        print("\nGetting predictions from all submodels...")

    all_predictions = np.zeros((n_samples, n_models))
    for i, model_name in enumerate(model_names):
        try:
            preds = predict_with_submodel_flexible(X_data, submodels[model_name], model_name, debug=debug)
            all_predictions[:, i] = preds
            if debug:
                print(f"  {model_name} prediction range: [{np.min(preds):.4f}, {np.max(preds):.4f}]")
        except Exception as e:
            if debug:
                print(f"  Error with {model_name}: {e}")
            all_predictions[:, i] = 0

    # Calculate distance-based weights
    if debug:
        print("\nCalculating distance-based weights...")

    weights = calculate_distance_weights(X_data, debug=debug)

    # Create weighted predictions
    weighted_predictions = all_predictions * weights

    if debug:
        print(f"\nFeature statistics:")
        print(f"  Raw predictions shape: {all_predictions.shape}")
        print(f"  Weighted predictions shape: {weighted_predictions.shape}")

    return {
        'predictions': all_predictions,
        'weights': weights,
        'weighted_predictions': weighted_predictions
    }

def calculate_distance_weights(X_data, debug=True):
    """Calculate distance-based weights for each model"""
    n_samples = X_data.shape[0]
    n_models = len(submodels)
    model_names = list(submodels.keys())

    # Initialize array for weights
    weights = np.ones((n_samples, n_models))

    # Define possible coordinate column pairs to check
    coord_cols = ['0', '1']  # As set in the dataset of PCA

    try:
        for i, model_name in enumerate(model_names):
            # Get cluster ID
            cluster_id = cluster_ids.get(model_name, i)

            # Get centroid
            centroid_idx = int(cluster_id) % len(centroids)
            centroid = centroids[centroid_idx]

            # Make sure centroid is the right shape for comparison
            if len(centroid) != 2:
                if debug:
                    print(f"  Centroid for {model_name} has {len(centroid)} dimensions, need 2. Using first 2.")
                centroid = centroid[:2]

            # Calculate distances
            try:
                # Extract coordinates
                coords = X_data[coord_cols].values

                # Calculate Euclidean distances efficiently
                diffs = coords - centroid
                distances = np.sqrt(np.sum(diffs**2, axis=1))

                # Calculate weights 
                weights[:, i] = 1 / (1 + np.exp(1 * distances))
                # weights[:, i] = 1 / ((1 + distances))**2

                if debug and i == 0:  # Show only for first model to reduce output
                    print(f"  Distance range for {model_name}: [{np.min(distances):.4f}, {np.max(distances):.4f}]")
                    print(f"  Weight range for {model_name}: [{np.min(weights[:, i]):.4f}, {np.max(weights[:, i]):.4f}]")

            except Exception as e:
                if debug:
                    print(f"  Error calculating weights for {model_name}: {e}")
                # Keep default weight
    except Exception as e:
        if debug:
            print(f"  Error in weight calculation: {e}. Using default weights.")

    return weights    

def train_rf_meta_model_with_cv(X_data, y_data, feature_mode='all', n_estimators=100, max_depth=None, 
                                cv_folds=3, debug=True):

    feature_dict = get_meta_model_features_flexible (X_data, debug=debug)

    if feature_mode == 'raw_only':
        meta_features = feature_dict['predictions']
        feature_names = [f'pred_{name}' for name in submodels.keys()]
    elif feature_mode == 'weighted_only':
        meta_features = feature_dict['weighted_predictions']
        feature_names = [f'weighted_{name}' for name in submodels.keys()]
    else:
        # Default to raw_only if an invalid mode is provided
        meta_features = feature_dict['predictions']
        feature_names = [f'pred_{name}' for name in submodels.keys()]
        if debug:
            print(f"  Warning: Unknown feature mode '{feature_mode}', defaulting to 'raw_only'")

    if debug:
        print(f"  Selected {len(feature_names)} features: {feature_names}")
        print(f"  Meta-features shape: {meta_features.shape}")

    # Normalize features
    feature_scaler = StandardScaler()
    scaled_features = feature_scaler.fit_transform(meta_features)

    if debug:
        print(f"  Scaled features range: [{np.min(scaled_features):.4f}, {np.max(scaled_features):.4f}]")

    # Create Random Forest model
    meta_model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42  # For reproducibility
    )

    # Perform cross-validation (silent - only collect metrics)
    kfold = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Cross-validation scores
    cv_r2_scores = cross_val_score(meta_model, scaled_features, y_data, cv=kfold, scoring='r2')
    cv_rmse_scores = -cross_val_score(meta_model, scaled_features, y_data, cv=kfold, 
                                     scoring='neg_root_mean_squared_error')

    # Store CV results (only for hyperparameter selection)
    cv_results = {
        'r2_scores': cv_r2_scores,
        'rmse_scores': cv_rmse_scores,
        'r2_mean': cv_r2_scores.mean(),
        'r2_std': cv_r2_scores.std(),
        'rmse_mean': cv_rmse_scores.mean(),
        'rmse_std': cv_rmse_scores.std()
    }

    # Train final model on all data
    meta_model.fit(scaled_features, y_data)

    # Get feature importances
    if debug:
        print("\nRandom Forest feature importances:")
        for i, name in enumerate(feature_names):
            print(f"  {name}: {meta_model.feature_importances_[i]:.6f}")

    # Get fitted values on full dataset
    fitted_values = meta_model.predict(scaled_features)

    # Calculate full dataset metrics
    train_rmse = np.sqrt(mean_squared_error(y_data, fitted_values))
    train_r2 = r2_score(y_data, fitted_values)
    train_mae = mean_absolute_error(y_data, fitted_values)

    if debug:
        print(f"\nFinal model performance (trained on all data):")
        print(f"  Training R²: {train_r2:.4f}")
        print(f"  Training RMSE: {train_rmse:.4f}")
        print(f"  Training MAE: {train_mae:.4f}")

    # Create final model visualization
    if debug:
        create_final_model_visualization(y_data, fitted_values, feature_mode, 
                                       n_estimators, max_depth)

    # Add CV results to metadata
    model_metadata = {
        'model': meta_model,
        'feature_scaler': feature_scaler,
        'feature_mode': feature_mode,
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'feature_names': feature_names,
        'model_names': list(submodels.keys()),
        'cv_results': cv_results,
        'train_r2': train_r2,
        'train_rmse': train_rmse,
        'train_mae': train_mae
    }

    return model_metadata, fitted_values, feature_dict, cv_results

def create_final_model_visualization(y_true, fitted_values, feature_mode, 
                                   n_estimators, max_depth):
    """Create visualization for final model performance only"""

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle(f'Random Forest Meta-Model Final Performance\n(Mode: {feature_mode}, Trees: {n_estimators}, Depth: {max_depth})', 
                 fontsize=14, fontweight='bold')

    # Training fit (full dataset)
    ax.scatter(y_true, fitted_values, alpha=0.6, s=30, color='blue', label='Training Data')
    max_val = max(max(y_true.max(), fitted_values.max()), 100)
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect Fit')

    # Calculate metrics
    train_r2 = r2_score(y_true, fitted_values)
    train_rmse = np.sqrt(mean_squared_error(y_true, fitted_values))
    train_mae = mean_absolute_error(y_true, fitted_values)

    # Add metrics text
    metrics_text = f'R² = {train_r2:.4f}\nRMSE = {train_rmse:.4f}\nMAE = {train_mae:.4f}'
    ax.text(0.05, 0.95, metrics_text, 
             transform=ax.transAxes, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))

    ax.set_xlabel('Actual Values')
    ax.set_ylabel('Predicted Values')
    ax.set_title('Model Performance on Full Training Set')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f'rf_meta_model_{feature_mode}_n{n_estimators}_d{max_depth}_final.png', 
                dpi=300, bbox_inches='tight')
    plt.close()

def run_rf_hyperparameter_search(data_file, output_file_prefix='rf_meta_model_results', 
                                n_estimators_list=[10, 50, 100], 
                                max_depth_list=[2, 4, 6, 8, 10, 20], 
                                cv_folds=3, debug=True):
    """
    Find best hyperparameters using cross-validation, then train final model
    """

    # Load data
    if debug:
        print(f"Loading data from {data_file}...")

    data = pd.read_csv(data_file)

    if debug:
        print(f"Data shape: {data.shape}")
        print(f"Data columns: {data.columns.tolist()}")

    # Assume 'IE' is your target column - replace with actual target
    target_column = 'IE'

    if target_column in data.columns:
        X_data = data.drop(target_column, axis=1)
        y_data = data[target_column]
        if debug:
            print(f"Target column found: {target_column}")
            print(f"Target statistics: mean={y_data.mean():.3f}, std={y_data.std():.3f}")
    else:
        raise ValueError(f"Target column '{target_column}' not found in data!")

    # Try different feature modes
    feature_modes = ['raw_only', 'weighted_only']
    best_results = {}

    print(f"\n{'='*70}")
    print(f"HYPERPARAMETER SEARCH WITH {cv_folds}-FOLD CROSS-VALIDATION")
    print(f"{'='*70}")
    print(f"Testing {len(n_estimators_list)} n_estimators values: {n_estimators_list}")
    print(f"Testing {len(max_depth_list)} max_depth values: {max_depth_list}")
    print(f"Total combinations per mode: {len(n_estimators_list) * len(max_depth_list)}")

    for mode in feature_modes:
        print(f"\n{'='*60}")
        print(f"SEARCHING BEST PARAMETERS FOR MODE: {mode}")
        print(f"{'='*60}")

        best_cv_r2 = -float('inf')
        best_params = {}
        all_results = []

        # Grid search with cross-validation (silent mode)
        for n_estimators in n_estimators_list:
            for max_depth in max_depth_list:
                # Train with current hyperparameters (silent)
                metadata, fitted, features, cv_results = train_rf_meta_model_with_cv(
                    X_data, y_data, feature_mode=mode, 
                    n_estimators=n_estimators, max_depth=max_depth,
                    cv_folds=cv_folds, debug=False
                )

                # Store results
                result = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth,
                    'cv_r2_mean': cv_results['r2_mean'],
                    'cv_r2_std': cv_results['r2_std'],
                    'cv_rmse_mean': cv_results['rmse_mean'],
                    'cv_rmse_std': cv_results['rmse_std'],
                    'train_r2': metadata['train_r2']
                }
                all_results.append(result)

                # Update best if better
                if cv_results['r2_mean'] > best_cv_r2:
                    best_cv_r2 = cv_results['r2_mean']
                    best_params = {
                        'n_estimators': n_estimators,
                        'max_depth': max_depth,
                        'cv_r2_mean': cv_results['r2_mean'],
                        'cv_r2_std': cv_results['r2_std'],
                        'cv_rmse_mean': cv_results['rmse_mean'],
                        'cv_rmse_std': cv_results['rmse_std']
                    }

        # Display top 5 results
        sorted_results = sorted(all_results, key=lambda x: x['cv_r2_mean'], reverse=True)
        print(f"\nTop 5 configurations for {mode}:")
        print(f"{'Rank':<5} {'n_trees':<8} {'max_depth':<10} {'CV R²':<15} {'CV RMSE':<15}")
        print("-" * 60)
        for i, res in enumerate(sorted_results[:5]):
            depth_str = str(res['max_depth']) if res['max_depth'] is not None else 'None'
            print(f"{i+1:<5} {res['n_estimators']:<8} {depth_str:<10} "
                  f"{res['cv_r2_mean']:.4f} ± {res['cv_r2_std']:.4f}  "
                  f"{res['cv_rmse_mean']:.4f} ± {res['cv_rmse_std']:.4f}")

        print(f"BEST {mode.upper()} CONFIGURATION:")
        print(f"   n_estimators = {best_params['n_estimators']}")
        print(f"   max_depth = {best_params['max_depth']}")
        print(f"   CV R² = {best_params['cv_r2_mean']:.4f} ± {best_params['cv_r2_std']:.4f}")
        print(f"   CV RMSE = {best_params['cv_rmse_mean']:.4f} ± {best_params['cv_rmse_std']:.4f}")

        # Train final model with best parameters and full output
        print(f"\n{'='*60}")
        print(f"TRAINING FINAL {mode.upper()} MODEL WITH BEST PARAMETERS")
        print(f"{'='*60}")

        metadata, fitted_values, feature_dict, cv_results = train_rf_meta_model_with_cv(
            X_data, y_data, feature_mode=mode, 
            n_estimators=best_params['n_estimators'], 
            max_depth=best_params['max_depth'],
            cv_folds=cv_folds, debug=True
        )

        # Create detailed results DataFrame
        results = pd.DataFrame({
            'actual': y_data,
            'fitted': fitted_values,
            'residual': y_data - fitted_values,
            'abs_residual': np.abs(y_data - fitted_values),
            'pct_error': 100 * (y_data - fitted_values) / y_data
        })
        rf_meta_model_weighted_only_n50_d6_best
        # Add individual model predictions
        for i, model_name in enumerate(metadata['model_names']):
            results[f'pred_{model_name}'] = feature_dict['predictions'][:, i]
            results[f'weighted_{model_name}'] = feature_dict['weighted_predictions'][:, i]

        # Save results
        depth_str = f"d{best_params['max_depth']}" if best_params['max_depth'] is not None else "dNone"
        output_file = f"{output_file_prefix}_{mode}_n{best_params['n_estimators']}_{depth_str}_best.csv"
        output = pd.concat([data, results.drop('actual', axis=1)], axis=1)
        output.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")

        # Save model
        model_file = f'rf_meta_model_{mode}_n{best_params["n_estimators"]}_{depth_str}_best.joblib'
        joblib.dump(metadata, model_file)
        print(f"Model saved to: {model_file}")

        # Store results
        best_results[mode] = {
            'best_params': best_params,
            'metadata': metadata,
            'results': results,
            'all_results': sorted_results
        }

    # Final comparison
    print(f"\n{'='*70}")
    print("FINAL COMPARISON OF BEST MODELS")
    print(f"{'='*70}")
    print(f"\n{'Mode':<15} {'n_trees':<8} {'max_depth':<10} {'CV R²':<20} {'Train R²':<10}")
    print("-" * 70)

    for mode, res in best_results.items():
        params = res['best_params']
        depth_str = str(params['max_depth']) if params['max_depth'] is not None else 'None'
        print(f"{mode:<15} {params['n_estimators']:<8} {depth_str:<10} "
              f"{params['cv_r2_mean']:.4f} ± {params['cv_r2_std']:.4f}   "
              f"{res['metadata']['train_r2']:.4f}")

    # Determine overall best model
    best_mode = max(best_results.items(), 
                    key=lambda x: x[1]['best_params']['cv_r2_mean'])[0]
    print(f"OVERALL BEST MODEL: {best_mode}")

    return best_results

