#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import joblib
import pandas as pd
import contextlib
import feature_engineering as fe  # Ensure this module is accessible

def perform_feature_engineering(file_path, file_name, target, objective, no_of_top_features=5):
    """
    Performs feature selection and engineering based on statistical tests.
    
    Parameters:
    - file_path (str): Path to the directory where the files are located.
    - file_name (str): Identifier for the data file.
    - target (str): Target variable for analysis.
    - objective (str): Type of problem ('classification' or 'regression').
    - no_of_top_features (int): Number of top features to select in each test.

    Returns:
    - df (DataFrame): DataFrame with engineered features.
    - new_cols (list): List of newly engineered feature columns.
    """
    
    print("\n******************\nperform_feature_engineering Function Running\n******************\n")
    
    # Define paths for saving and loading data
    path_to_save = f"{file_path}/pkl/{target}_results"
    path_to_file = f"{path_to_save}/df_train_{target}_scaled.pkl"
    path_to_test_file = f"{path_to_save}/df_test_{target}_scaled.pkl"
    path_to_features = f"{file_path}/pkl/{file_name}_feature_list.pkl"
    path_to_full_dataset_features = f"{file_path}/pkl/{file_name}_features.pkl"
    path_to_full_dataset_data = f"{file_path}/pkl/{file_name}_data.pkl"
    
    log_path = f"{path_to_save}/log/output_log_04_{__name__}.txt"
    with open(log_path, 'w') as f, contextlib.redirect_stdout(f):
    
        ### Top features from ANOVA Test
        df_anova = joblib.load(f"{path_to_save}/ANOVA_F_test_result_{target}_{objective}.pkl")
        anova_features = df_anova['feature_names'].tolist()[:no_of_top_features]
        print(f"\nTop {no_of_top_features} Features from Analysis of Variance Test:")
        print(df_anova.iloc[:no_of_top_features, :])
    
        ### Top features from Mutual Information Test
        df_mi = joblib.load(f"{path_to_save}/MI_result_{target}.pkl")
        mi_features = df_mi['feature_names'].tolist()[:no_of_top_features]
        print(f"\nTop {no_of_top_features} Features from Mutual Information Test:")
        print(df_mi.iloc[:no_of_top_features, :])
    
        # Initialize empty list for optional classification-specific features
        ld_features = []
    
        if objective == 'classification':
            # Top features from Logistic Discrimination Test
            df_ld = joblib.load(f"{path_to_save}/logistic_discrimination_result_{target}.pkl")
            df_ld = df_ld.sort_values('coefficient')
            ld_features = df_ld['feature_names'].tolist()[:no_of_top_features] + df_ld['feature_names'].tolist()[-no_of_top_features:]
            print(f"\nTop {no_of_top_features} Features from Logistic Discrimination:")
            print(df_ld.iloc[:no_of_top_features, :])
            print(f"\nBottom {no_of_top_features} Features from Logistic Discrimination:")
            print(df_ld.iloc[-no_of_top_features:, :])
    
        ### Top features from Gradient Boosted Feature Selection Test
        df_gbfs = joblib.load(f"{path_to_save}/feature_relevance_score_{target}.pkl")
        gbfs_features = df_gbfs['feature'].tolist()[:no_of_top_features]
        print(f"\nTop {no_of_top_features} Features from GBFS:")
        print(df_gbfs.iloc[:no_of_top_features, :])
    
        # Initialize feature engineering with selected features
        perform = fe.engineering(
            path_to_file=path_to_file,
            path_to_test_file=path_to_test_file,
            path_to_save=path_to_save,
            target=target,
            features=path_to_features
        )
    
        # Combine features based on objective type
        feature_list = anova_features + mi_features + gbfs_features + ld_features 
        print(f"\nTotal Number of Features: {len(feature_list)}")
    
        # Perform feature engineering
        print(f"\nPerforming Feature Engineering on {len(feature_list)} Features")
        df_train, df_test, new_cols = perform.brute_force(feature_list=feature_list)
        print("\nEngineered Feature Results (train):")
        print(df_train)
        print("\nEngineered Feature Results (test):")
        print(df_test)
    
        # Save engineered features
        perform.save()
    
        print("\nPerforming Feature Engineering on the full dataset")
        perform.brute_force_on_all(path_to_full_dataset_features=path_to_full_dataset_features, 
                                   path_to_full_dataset_data=path_to_full_dataset_data, 
                                   feature_list=feature_list)

    return df_train, df_test, new_cols
