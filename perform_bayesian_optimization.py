#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import optimization as opt
import joblib
import math
import contextlib

def round_down_to_one_sig_fig(number):
    if number == 0:
        return 0
    magnitude = 10 ** math.floor(math.log10(abs(number)))
    
    # Always round down by using floor for positive and ceil for negative
    if number > 0:
        return math.floor(number / magnitude) * magnitude
    else:
        return math.ceil(number / magnitude) * magnitude

def round_up_to_one_sig_fig(number):
    if number == 0:
        return 0
    magnitude = 10 ** math.floor(math.log10(abs(number)))
    
    # Always round up by using ceil for positive and floor for negative
    if number > 0:
        return math.ceil(number / magnitude) * magnitude
    else:
        return math.floor(number / magnitude) * magnitude


def perform_bayesian_optimization(file_path, file_name, target, objective, scaled=False, 
                                boosting_method='lightGBM', optimization_method='bayesian', n_calls=200,
                                strategy='weighted', adjusted=False):
    """
    Perform model optimization, training, and evaluation for regression or classification tasks.
    
    Parameters:
    - file_path (str): Path to the working directory where files are stored.
    - file_name (str): Identifier for the dataset.
    - target (str): Target variable for prediction.
    - objective (str): Type of prediction task ('classification' or 'regression').
    - file_name (str): Identifier for the dataset (default: 'dataset').
    - scaled (bool): Whether to scale features. Used only if `objective` is 'classification'.
    - boosting_method (str): Boosting method for the model. Default is 'lightGBM'.
    - optimization_method (str): Optimization method for hyperparameters (e.g., 'bayesian').
    - n_calls (int): Number of optimization calls for the hyperparameter search (default: 200).
    - strategy (str): Strategy for evaluation (default: 'weighted').
    - adjusted (bool): Whether to adjust predictions in evaluation.
    """

    print("\n******************\nperform_bayesian_optimization Function Running\n******************\n")

    # Define paths
    path_to_save = f"{file_path}/pkl/{target}_results"
    path_to_train_data = f"{path_to_save}/df_train_{target}_engineered.pkl"
    path_to_test_data = f"{path_to_save}/df_test_{target}_engineered.pkl"
    path_to_features = f"{path_to_save}/features_selected_from_RFE_{target}.pkl"
    
    log_path = f"{path_to_save}/log/output_log_08_{__name__}.txt"
    with open(log_path, 'w') as f, contextlib.redirect_stdout(f):
            
        # Load selected features
        features = joblib.load(path_to_features)
        print(f"Number of loaded features: {len(features)}")
    
        df_data = joblib.load(f"{file_path}/pkl/{file_name}_data.pkl")
    
        # Print target stats
        print(f"Target Variable: {target}")
    
        if objective == 'regression':
            max_target = df_data[target].max()
            max_rounded = round_up_to_one_sig_fig(max_target)
            min_target = df_data[target].min()
            min_rounded = round_down_to_one_sig_fig(min_target)
            print('Max value: ', max_target)
            print('Max value rounded: ', max_rounded)
            print('Min value: ', min_target)
            print('Min value rounded: ', min_rounded)
        
        print(f"Objective: {objective}")
            
        # Initialize optimization object
        final_stage = opt.optimization(
            path_to_train_data=path_to_train_data,
            path_to_test_data=path_to_test_data,
            path_to_save=path_to_save,
            features=path_to_features,
            target=target,
            problem=objective,
            scaled=scaled if objective == 'classification' else False  # Scaled only for classification
        )
    
        # Choose and set base model
        final_stage.base_model(boosting_method=boosting_method)
    
        # Define hyperparameters (ranges are edited within the function as per usage needs)
        final_stage.set_hyperparameters()
    
        # Run optimization with specified method
        final_stage.run(optimization_method=optimization_method, n_calls=n_calls)
    
        # Plot convergence of the optimization process
        final_stage.convergence_plot()
    
        # Objective plot to visualize results of the optimization
        final_stage.objective_plot()
    
        # Adjust objective plot based on sample source and minimum search criteria
        final_stage.objective_plot_adjust(
            sample_source='expected_minimum',
            minimum='expected_minimum',
            n_minimum_search=2
        )
    
        # Train model with the optimal hyperparameters
        final_stage.train_model()
    
        # Evaluate model on the test set
        if objective == 'regression':
            final_stage.evaluate(strategy=strategy, adjusted=adjusted, max_value=max_rounded, min_value=min_rounded)
        
        # Additional evaluation plots for classification tasks
        if objective == 'classification':
            
            final_stage.evaluate(strategy=strategy, adjusted=adjusted, target_names=['False', 'True'])
    
            # Plot ROC curve
            final_stage.ROC(overall_performance=True, positive_class=1)
    
            # Plot DET curve
            final_stage.DET()
    
            # Plot Precision-Recall (PR) curve
            final_stage.PR(positive_class=1)