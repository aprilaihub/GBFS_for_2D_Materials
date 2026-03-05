#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import joblib
import contextlib

import multicollinearity_reduction as mr  # Ensure this module is accessible

def perform_multicollinearity_reduction(file_path, file_name, target, objective, no_of_relevant_features=200, correlation_threshold=0.9, max_link_threshold = 5):
    """
    Reduces multicollinearity in features based on correlation and hierarchical clustering.
    
    Parameters:
    - file_path (str): Path to the directory where files are stored.
    - file_name (str): Identifier for the dataset.
    - target (str): Target variable for analysis.
    - objective (str): Type of analysis objective ('classification' or 'regression').
    - no_of_relevant_features (int): Number of top relevant features to select from GBFS. (Default is 200)
    - correlation_threshold (float): Threshold for correlation to remove highly correlated features. (Default is 0.9)
    - max_link_threshold (int): Cap maximum threshold to prevent excessive computations. (Default is 5)

    Returns:
    - final_features (dict): Dictionary of selected features at each threshold level.
    """
    
    print("\n******************\nperform_multicollinearity_reduction Function Running\n******************\n")
    
    # Define paths for saving and loading data
    path_to_save = f"{file_path}/pkl/{target}_results"
    path_to_file = f"{path_to_save}/df_train_{target}_engineered.pkl"
    
    log_path = f"{path_to_save}/log/output_log_05_{__name__}.txt"
    with open(log_path, 'w') as f, contextlib.redirect_stdout(f):
    
        # Select top n features from GBFS result
        gbfs_cols = joblib.load(f"{path_to_save}/feature_relevance_score_{target}.pkl")['feature'].tolist()[:no_of_relevant_features]
        new_cols = joblib.load(f"{path_to_save}/features_{target}_engineered.pkl")
        features = gbfs_cols + new_cols
    
        print('No. of columns from GBFS:', len(gbfs_cols))
        print('No. of new columns:', len(new_cols))
        print('No. of Features:', len(features))
        
        # Initialize the multicollinearity reduction process
        data = mr.multicollinearity_reduction(
            path_to_file=path_to_file,
            path_to_save=path_to_save,
            target=target,
            features=features
        )
    
        # Remove features with correlation coefficient greater than the specified threshold
        data.correlation_analysis(threshold=correlation_threshold)
        data.apply_correlation_filter()
    
        # Perform hierarchical clustering analysis and create dendrogram
        data.hierarchical_cluster_analysis()
    
        # Generate dynamic threshold list based on number of links created
        num_links = len(features) - 1  # Maximum possible number of links
        threshold_list = list(range(1, min(max_link_threshold, num_links + 1)))
    
        final_features = {}
        
        for i in threshold_list:
            selected_features = data.apply_linkage_threshold(threshold=i)
            final_features[i] = selected_features
            print(f"Number of features selected with linkage threshold {i}: {len(selected_features)}")
            
    return final_features
