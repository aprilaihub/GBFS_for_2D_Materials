#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import feature_analyses as fa
import contextlib

def perform_feature_analysis(file_path, file_name, target, objective):
    """
    Performs feature analysis on a given dataset by executing various statistical tests.
    
    Parameters:
    - file_path (str): Path to the directory where the files are located.
    - file_name (str): Identifier for the data file.
    - target (str): Target variable for analysis.
    - objective (str): Type of problem ('classification' or 'regression').

    Returns:
    - df (DataFrame): Data with constant features removed.
    - df_anova (DataFrame): Results of ANOVA F-test.
    - df_mi (DataFrame): Results of mutual information test.
    - df_chi2 (DataFrame or None): Results of Chi-Squared test if 'classification', otherwise None.
    - df_ld (DataFrame or None): Results of logistic discrimination test if 'classification', otherwise None.
    """
    
    print("\n******************\nperform_feature_analysis Function Running\n******************\n")
    
    # Paths for data and features
    path_to_save = f"{file_path}/pkl/{target}_results"
    path_to_file = f"{path_to_save}/df_train_{target}_scaled.pkl"
    path_to_features = f"{file_path}/pkl/{file_name}_feature_list.pkl"
    
    # # Redirect stdout to a file
    # with open(f"{path_to_save}/log/output_log_03_{__name__}.txt", 'w') as f:
    #     original_stdout = sys.stdout
    #     sys.stdout = f
    log_path = f"{path_to_save}/log/output_log_03_{__name__}.txt"
    with open(log_path, 'w') as f, contextlib.redirect_stdout(f):
    
        # Initialize analysis
        analyses = fa.perform(
            path_to_file=path_to_file,
            path_to_save=path_to_save,
            target=target,
            features=path_to_features
        )
    
        # Remove features which remain constant (numerical features only)
        df = analyses.remove_constant_features()
        print("Data after removing constant features:")
        print(df.head())
    
        # Perform Analysis of Variance (ANOVA) Test
        df_anova = analyses.ANOVA_F_test(problem=objective, all_features=False, csv=False)
        print("ANOVA F-Test Results:")
        print(df_anova)
        
        # Perform Mutual Information Test
        df_mi = analyses.mutual_information(problem=objective, csv=False)
        print("Mutual Information Test Results:")
        print(df_mi)
    
        # Initialize variables for optional classification tests
        df_chi2, df_ld = None, None
    
        if objective == 'classification':
            # # Perform Chi-Squared Analysis (Only for categorical features)
            # df_chi2 = analyses.chi2_test(csv=False)
            # print("Chi-Squared Test Results:")
            # print(df_chi2)
    
            # Perform Logistic Discrimination Test
            df_ld = analyses.logistic_discrimination(
                target_classes=2,
                class_names=['Metal', 'Non-Metal'],
                csv=False
            )
            print("Logistic Discrimination Test Results:")
            print(df_ld)
       
    return df, df_anova, df_mi, df_chi2, df_ld
