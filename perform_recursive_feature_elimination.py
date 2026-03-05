#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import joblib
import contextlib
import recursive_feature_elimination as rfe

def perform_recursive_feature_elimination(file_path, file_name, target, objective, threshold=2, cv_fold=10, boosting_method='lightGBM'):
    """
    Performs Recursive Feature Elimination (RFE) for feature selection based on a specified hierarchical threshold value.
    
    Parameters:
    - file_path (str): Path to the directory where files are stored.
    - file_name (str): Identifier for the dataset.
    - target (str): Target variable for analysis.
    - objective (str): Type of analysis objective ('classification' or 'regression').
    - threshold (int): Specific threshold level for selecting features. (Default is 2)
    - cv_fold (int): Number of cross-validation folds for RFE. (Default is 10)
    - boosting_method (str): Boosting method to use for the base model in RFE. (Default is 'lightGBM')
    
    Returns:
    - RFE_features (list): List of selected features after RFE based on the specified threshold.
    """
    
    print("\n******************\nperform_recursive_feature_elimination Function Running\n******************\n")
    
    # Define path to results directory
    path_to_save = f"{file_path}/pkl/{target}_results"
    path_to_file = f"{path_to_save}/df_train_{target}_engineered.pkl"
    
    log_path = f"{path_to_save}/log/output_log_06_{__name__}.txt"
    with open(log_path, 'w') as f, contextlib.redirect_stdout(f):
    
        # Load features selected from hierarchical analysis for the specified threshold
        features_file = f"{path_to_save}/features_selected_from_hierarchical_analysis_{target}_threshold_{threshold}.pkl"
        try:
            features = joblib.load(features_file)
            print(f"Loaded {len(features)} features for threshold {threshold}")
        except FileNotFoundError:
            print(f"No features file found for threshold {threshold}. Please check the specified threshold or file availability.")
            return None
        
        # Initialize the recursive feature elimination process
        run = rfe.recursive_feature_elimination(
            path_to_file=path_to_file,
            path_to_save=path_to_save,
            target=target,
            features=features,
            scaled=False,  # Scaling parameter can be modified if needed
            problem=objective
        )
        
        # Set up and run RFE
        run.base_model(boosting_method=boosting_method) # estimator = 
        RFE_features = run.perform(cv_fold=cv_fold)
        
        # Plot RFE results for the specified threshold
        run.RFE_plot()
        print(f"RFE completed for threshold {threshold}. Selected features: {len(RFE_features)}")

    return RFE_features