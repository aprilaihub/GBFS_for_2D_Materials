#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import joblib
import contextlib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import data_manipulation


###############################################################################

from sklearn.neighbors import NearestNeighbors
import numpy as np

def smooth_target_knn(df_train, feature_list, target, k=5):
    """
    Smooths the target column in df_train using K-Nearest Neighbors averaging.
    
    Parameters:
    - df_train (DataFrame): Training set.
    - feature_list (list): List of feature column names.
    - target (str): Name of the target variable.
    - k (int): Number of neighbors to use for smoothing. Default is 5.
    
    Returns:
    - df_train (DataFrame): Modified training set with a new column 'target_smoothed'.
    """
    X = df_train[feature_list].values
    y = df_train[target].values

    knn = NearestNeighbors(n_neighbors=k)
    knn.fit(X)
    _, indices = knn.kneighbors(X)

    smoothed_target = np.array([np.mean(y[neigh]) for neigh in indices])
    df_train[f"{target}_smoothed"] = smoothed_target

    return df_train

###############################################################################

### Original Section


def split_scale_data(file_path, file_name, target, objective, target_filter=None, features_filter=None, dataset_filter=None, stability_filter=False, metallicity_filter=False, test_size=0.2, path_to_expt=None):
    """
    Preprocesses data for machine learning by loading, cleaning, filtering features, 
    filtering based on the target variable, splitting into train and test sets, scaling,
    and optionally incorporating experimental data into the test set.
    
    SPECIAL CASE: If target == "is_metal" create the "is_metal" column and populate it based on the "band_gap_dir" column

    Parameters:
    - file_path (str): Base file path where data and results are stored.
    - file_name (str): Base name of the data file (without extension).
    - target (str): Target variable name.
    - objective (str): The type of learning task, e.g., 'classification' or 'regression'.
    - target_filter (function): Function to filter target variable rows (e.g., lambda x: x > 0.2). Default is None.
    - test_size (float): Proportion of data in test dataset. Default value is 0.2.
    - path_to_expt (str): Path to the experimental data directory. Default is None.

    Returns:
    - df_train (DataFrame): Processed training set.
    - df_test (DataFrame): Processed test set (includes experimental data if provided).
    - feature_list (list): List of filtered feature column names.

    Raises:
    - FileNotFoundError: If any required data or feature file is missing.
    """
    
    print("\n******************\nsplit_scale_data Function Running\n******************\n")
    
    # File names and paths
    data_file_name = f"{file_name}_data"
    data_file_path = f"{file_path}/pkl/{data_file_name}.pkl"
    features_file_name = f"{file_name}_features"
    features_file_path = f"{file_path}/pkl/{features_file_name}.pkl"
    results_path = f"{file_path}/pkl/{target}_results/"
        
    # Create results directory if it doesn't exist
    os.makedirs(results_path, exist_ok=True)
    
    # Create log directory if it doesn't exist
    os.makedirs(f"{results_path}log/", exist_ok=True)
    
    # Redirect print statements to a file
    log_path = f"{results_path}log/output_log_01_{__name__}.txt"
    with open(log_path, 'w') as f, contextlib.redirect_stdout(f):
    
        # Load and clean data file with error handling
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"The data file '{data_file_path}' does not exist.")
        with open(data_file_path, 'rb') as f:
            data = joblib.load(f).infer_objects(copy=False)
        print(f"\nThe data file '{data_file_name}' has been loaded successfully from '{data_file_path}'\n")
        
        # Load and preprocess features file with error handling
        if not os.path.exists(features_file_path):
            raise FileNotFoundError(f"The features file '{features_file_path}' does not exist.")
        with open(features_file_path, 'rb') as f:
            features = joblib.load(f).infer_objects(copy=False)
        print(f"\nThe features file '{features_file_name}' has been loaded successfully from '{features_file_path}'\n")
        
        # Filter data and features based on whether the dataset source is in dataset_filter     
        if dataset_filter is not None:
            print(f"Remove data from {dataset_filter}")
            features = features[~data['source'].isin(dataset_filter)]
            data = data[~data['source'].isin(dataset_filter)]
            print(f"Using data from {data['source'].unique()}")
            print(f"Total number of entries: {len(data)}")
                                   
        # Filter data and features based on the 'is_stable' flag
        if stability_filter: 
            features = features[data['is_stable']]
            data = data[data['is_stable']]
            
        # Filter data and features based on the 'is_metal' flag. If non-metal then keep
        if metallicity_filter:
            features = features[~data['is_metal']]
            data = data[~data['is_metal']]
            
        # Exclude text-based entries and save as {file_name}_features_fillna.pkl   
        features = features.select_dtypes(exclude=['object', 'string']).fillna(0)
        joblib.dump(features, f"{file_path}/pkl/{file_name}_features_fillna.pkl")
        
        # Data manipulation to remove empty columns and columns with single entries.
        dm = data_manipulation.data_manipulation(f"{file_path}/pkl/{file_name}_features_fillna.pkl") 
        single_entry_column_list = dm.single_entry_col()
        features = dm.drop_col(single_entry_column_list)
        empty_column_list = dm.empty_col()
        features = dm.drop_col(empty_column_list)
        if 'compound possible' in features.columns:
            features = dm.drop_col('compound possible')
        if features_filter is not None:
            features = dm.drop_col(features_filter)
        
        
        # Feature list
        feature_list = features.columns.tolist()
        
        # Save feature list and filtered features
        joblib.dump(feature_list, f"{file_path}/pkl/{file_name}_feature_list.pkl")
        joblib.dump(features, f"{file_path}/pkl/{file_name}_features_filtered.pkl")
        
        # Add "is_metal" column for whether the band_gap_dir is below 0.2 eV
        if target == "is_metal":
            if 'is_metal' not in data.columns:
                data['is_metal'] = data["band_gap_dir"] < 0.2
            # Counting True and False entries
            true_count = data['is_metal'].sum()  # True values count
            false_count = len(data) - true_count    # False values count
            
            # Display the results
            print(f"Number of Metal entries: {true_count}")
            print(f"Number of Non-Metal entries: {false_count}")
            
        # Add "is_stable" column for whether the e_above_hull is below 0.05 eV
        if target == "is_stable":
            if 'is_stable' not in data.columns:
                data['is_stable'] = data['e_above_hull'] < 0.05
            # Counting True and False entries
            true_count = data['is_stable'].sum()  # True values count
            false_count = len(data) - true_count    # False values count
            
            # Display the results
            print(f"Number of Stable entries: {true_count}")
            print(f"Number of Unstable entries: {false_count}")
            
        if target == "is_gap_direct":            
            true_count = data['is_gap_direct'].sum()  # True values count
            false_count = len(data) - true_count    # False values count
            
            
        # Combine target column and features into pandas.DataFrame
        df = pd.concat([data[target], features], axis=1)
        
        # Add 'is_experimental' flag for non-experimental data
        df['is_experimental'] = 0.0
        columns_order = [target, 'is_experimental'] + [col for col in df.columns if col not in [target, 'is_experimental']]
        df = df[columns_order]
         
        # Apply target filter if specified
        if target_filter is not None:
            df = df[df[target].apply(target_filter)]
        
        # Split dataset
        if objective == 'classification':
            df_train, df_test = train_test_split(df, test_size=test_size, stratify=df[target].tolist(), random_state=42)  # random_state=42
        elif objective == 'regression':
            df_train, df_test = train_test_split(df, test_size=test_size, random_state=42) # random_state=42
        
# ###############################################################################
        
#         if objective == 'regression':
#             df_train = smooth_target_knn(df_train, feature_list, target, k=5)
#             # You can now choose to use df_train[f"{target}_smoothed"] instead of df_train[target] in your regression model
        
# ###############################################################################
        
        # Incorporate experimental data into test set if path is provided
        if path_to_expt is not None:
            # Define paths for experimental data and features
            expt_features_path = f"{path_to_expt}/expt_{target}_features_scaled.pkl"
            expt_data_path = f"{path_to_expt}/expt_{target}_obj_data.pkl"
            
            # Load experimental features
            if not os.path.exists(expt_features_path):
                raise FileNotFoundError(f"The experimental features file '{expt_features_path}' does not exist.")
            with open(expt_features_path, 'rb') as f:
                expt_features = joblib.load(f).infer_objects(copy=False)
            
            # Load experimental data
            if not os.path.exists(expt_data_path):
                raise FileNotFoundError(f"The experimental data file '{expt_data_path}' does not exist.")
            with open(expt_data_path, 'rb') as f:
                expt_data = joblib.load(f).infer_objects(copy=False)
            
            # Combine experimental data with features
            expt_data = pd.concat([expt_data[target], expt_features], axis=1)
            
            # Ensure only features present in the main dataset are retained
            retained_columns = [target] + feature_list  # Retain target column and feature list
            expt_data = expt_data[expt_data.columns.intersection(retained_columns)]
            
            # Add experimental flag
            expt_data['is_experimental'] = 1.0
            expt_data = expt_data[[target, 'is_experimental'] + [col for col in expt_data.columns if col not in [target, 'is_experimental']]]
        
            # Add experimental data to the test set
            df_test = pd.concat([df_test, expt_data], ignore_index=True)
            print(f"Experimental data has been successfully loaded from '{path_to_expt}' and added to the test set.")
        
        # Save unscaled data
        joblib.dump(df_train, f"{results_path}df_train_{target}.pkl")
        print(f"Unscaled training data saved to {results_path}df_train_{target}.pkl")
        joblib.dump(df_test, f"{results_path}df_test_{target}.pkl")
        print(f"Unscaled test data saved to {results_path}df_test_{target}.pkl")
        
        
        # Scale training set
        scaling = MinMaxScaler(feature_range=(0, 1))
        df_train[feature_list] = pd.DataFrame(
            scaling.fit_transform(df_train[feature_list].values),
            columns=df_train[feature_list].columns,
            index=df_train[feature_list].index
        )
        # Scale test set
        df_test[feature_list] = pd.DataFrame(
            scaling.transform(df_test[feature_list].values),
            columns=df_test[feature_list].columns,
            index=df_test[feature_list].index
        )
        
        # Save scaled data
        joblib.dump(df_train, f"{results_path}df_train_{target}_scaled.pkl")
        print(f"Scaled training data saved to {results_path}df_train_{target}_scaled.pkl")
        joblib.dump(df_test, f"{results_path}df_test_{target}_scaled.pkl")
        print(f"Scaled test data saved to {results_path}df_test_{target}_scaled.pkl")
    
    return df_train, df_test, feature_list