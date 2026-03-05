#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
file_path = os.path.abspath('.')
import sys

import warnings
warnings.filterwarnings("ignore")
import contextlib

import GBFS as gb

import joblib
import pandas as pd


def perform_GBFS(file_path, file_name, target, objective, cv_folds=10, oversampled = False):
    """
    Takes scaled features in the training set and performs gradient-boosted feature selection, 
    followed by feature relevance ranking and recursive feature selection. 
    

    Parameters:
    - file_path (str): Base file path where data and results are stored.
    - file_name (str): Base name of the data file (without extension).
    - target (str): Target variable name.
    - objective (str): regression or classification mode
    - cv_folds (int): Number of folds used in k-fold cross-validation. Default value is 10.
    - oversampled (bool): Option to use oversampling (classification only). Default value is False.

    Returns:
    - None

    Raises:
    - FileNotFoundError: If any required data or feature file is missing.
    """
    
    print("\n******************\nperform_GBFS Function Running\n******************\n")
    
    # File names and paths
    data_file_name = f"{file_name}_data"
    data_file_path = f"{file_path}/pkl/{data_file_name}.pkl"
    
    results_path = f"{file_path}/pkl/{target}_results/"
    
    features_file_name = f"df_train_{target}_scaled"  
    features_file_path = f"{results_path}/{features_file_name}.pkl"
    
    feature_list_file_name = f"{file_name}_feature_list"
    feature_list_file_path = f"{file_path}/pkl/{feature_list_file_name}.pkl"

    # Redirect print statements to a file
    log_path = f"{results_path}/log/output_log_02_{__name__}.txt"
    with open(log_path, 'w') as f, contextlib.redirect_stdout(f):
        
        # Conduct grid search to perform preliminary scan of the hyperparameter space & retrieve a trained model
        if objective == 'regression':
            # GBFS Initialization 
            FS = gb.GBFS(
                        path_to_file = features_file_path, 
                        path_to_save = results_path, 
                        target = target,
                        features = feature_list_file_path,
                        oversampled_it = False, 
                        problem = 'regression'
                        )
            # GBFS Run
            FS.run(
                    boosting_method = 'lightGBM', 
                    cv_folds = cv_folds
                    )
            # Obtain feature relevance score
            FS.feature_relevance(
                                plot = True, 
                                no_of_features = 20
                                )
            
            # Perform recursive feature selection 
            FS.recursive_selection(
                                stratify = False, 
                                oversample_technique = None, 
                                chosen_metric = 'rmse', # Choose: 'f1_score', 'accuracy', 'balanced_accuracy', 'hamming_loss', 'roc_auc', 'average_precision'
                                average = 'weighted',
                                no_to_terminate = 200,
                                max_no_imp = 10
                                )
            
            # Plot the result
            FS.convergence_plot()
            
            # 'train_mae', 'va_mae', 
            # 'train_mse', 'va_mse', 
            # 'train_rmse', 'va_rmse', 
            # 'train_r_sq', 'va_r_sq', 
            # 'train_max_error', 'va_max_error'
            
            # Plot the a subset of the results
            FS.convergence_plot(
                                train_metric = 'train_mae',
                                validation_metric = 'va_mae'
                                )
            
            # Plot the result
            FS.convergence_plot(
                                train_metric = 'train_r_sq',
                                validation_metric = 'va_r_sq'
                                )
            
            # Plot the result
            FS.convergence_plot(
                                train_metric = ['train_r_sq', 'train_rmse', 'train_mae'],
                                validation_metric = ['va_r_sq', 'va_rmse', 'va_mae']
                                )
            
        elif objective == 'classification':
            # GBFS Initialization 
            FS = gb.GBFS(
                        path_to_file = features_file_path, 
                        path_to_save = results_path, 
                        target = target,
                        features = feature_list_file_path,
                        oversampled_it = oversampled, 
                        problem = 'classification',
                        target_classes=2
                        )
            # GBFS run
            FS.run(
                    boosting_method = 'lightGBM', 
                    objective='binary',
                    oversample_technique='smoothed_ros', 
                    cv_folds = cv_folds
                    )
            
            # Obtain feature relevance score
            FS.feature_relevance(
                                plot = True, 
                                no_of_features = 20
                                )
            
            # Perform recursive feature selection 
            FS.recursive_selection(
                                stratify = False, 
                                oversample_technique = None, 
                                chosen_metric = 'f1_score', # Choose: 'f1_score', 'accuracy', 'balanced_accuracy', 'hamming_loss', 'roc_auc', 'average_precision'
                                average = 'weighted',
                                no_to_terminate = 200,
                                max_no_imp = 10
                                )
            
            # Plot the result
            FS.convergence_plot()
            
            # 'train_acc', 'validation_acc',
            # 'train_b_acc', 'validation_b_acc', 
            # 'train_hamming', 'validation_hamming', 
            # 'train_avg_precision', 'validation_avg_precision', 
            # 'train_roc_auc', 'validation_roc_auc', 
            # 'train_f1', 'validation_f1'
                                           
            # Plot the a subset of the results
            FS.convergence_plot(
                                train_metric = 'train_avg_precision',
                                validation_metric = 'validation_avg_precision'
                                )
            
            # Plot the result
            FS.convergence_plot(
                                train_metric = 'train_roc_auc',
                                validation_metric = 'validation_roc_auc'
                                )
            
            # Plot the result
            FS.convergence_plot(
                                train_metric = 'train_f1',
                                validation_metric = 'validation_f1'
                                )
            
            # Plot the result
            FS.convergence_plot(
                                train_metric = ['train_avg_precision', 'train_roc_auc', 'train_f1'],
                                validation_metric = ['validation_avg_precision', 'validation_roc_auc', 'validation_f1']
                                )
            # # Reset stdout
            # sys.stdout = original_stdout