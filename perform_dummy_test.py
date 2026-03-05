#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
import math
import joblib
import contextlib
import seaborn as sns
from lightgbm import LGBMRegressor, LGBMClassifier
from xgboost import XGBRegressor, XGBClassifier

import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import jaccard_score, multilabel_confusion_matrix, roc_curve, roc_auc_score, \
                            auc, f1_score, classification_report, recall_score, precision_recall_curve, \
                            balanced_accuracy_score, confusion_matrix, accuracy_score, average_precision_score, \
                            hamming_loss, matthews_corrcoef

import dummy_model as dm

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

def perform_dummy_test(file_path, file_name, target, objective, unit, threshold=2, 
                     learning_rate=0.1, n_estimators=350, num_leaves=40, 
                     max_depth=-1, random_state=42, no_of_features_to_plot=20,
                     model_type="lightGBM"):
    """
    Performs a LightGBM or XGBoost task (regression or classification), prediction, and feature importance analysis.
    
    Parameters:
    - file_path (str): Path to the directory where files are stored.
    - file_name (str): Identifier for the dataset.
    - target (str): Target variable for analysis.
    - objective (str): Type of analysis objective ('classification' or 'regression'). (Default is 'regression')
    - unit (str): Unit of the target variable.
    - threshold (int): Threshold level used for selecting features. (Default is 2)
    - learning_rate (float): Learning rate for the model. (Default is 0.1)
    - n_estimators (int): Number of boosting iterations. (Default is 350)
    - num_leaves (int): Maximum tree leaves for base learners (only for LightGBM). (Default is 40)
    - max_depth (int): Maximum tree depth. (-1 indicates no limit). (Default is -1)
    - random_state (int): Random seed. (Default is 42)
    - no_of_features_to_plot (int): Number of top features to plot for feature importance. (Default is 20)
    - model_type (str): Type of model to use ('lightGBM' or 'XGBoost'). (Default is 'lightGBM')
    
    Returns:
    - df_pred (DataFrame): DataFrame containing actual and predicted target values.
    - feature_score (DataFrame): DataFrame containing features and their relevance scores.
    """
    
    print("\n******************\nperform_dummy_test Function Running\n******************\n")
    
    # Define paths for training and test data
    path_to_save = f"{file_path}/pkl/{target}_results"
    path_to_train_file = f"{path_to_save}/df_train_{target}_engineered.pkl"
    path_to_test_file = f"{path_to_save}/df_test_{target}_engineered.pkl"
    
    log_path = f"{path_to_save}/log/output_log_07_{__name__}.txt"
    with open(log_path, 'w') as f, contextlib.redirect_stdout(f):
    
        # Load training and testing datasets
        df_train = joblib.load(path_to_train_file)
        df_test = joblib.load(path_to_test_file)
    
        # Load features selected from hierarchical analysis for the specified threshold
        features_file = f"{path_to_save}/features_selected_from_hierarchical_analysis_{target}_threshold_{threshold}.pkl"
        try:
            features = joblib.load(features_file)
            print(f"No. of Features Selected From Hierarchical Analysis: {len(features)}")
        except FileNotFoundError:
            print(f"No features file found for threshold {threshold}. Please check the specified threshold or file availability.")
            return None, None
        
        # Define model based on objective and model type
        if model_type == "lightGBM":
            if objective == 'classification':
                model = LGBMClassifier(
                    importance_type='gain',
                    objective='binary',
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    num_leaves=num_leaves,
                    max_depth=max_depth,
                    random_state=random_state
                )
            elif objective == 'regression':  # regression
                model = LGBMRegressor(
                    boosting_type='gbdt',
                    objective='regression',
                    importance_type='gain',
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    num_leaves=num_leaves,
                    max_depth=max_depth,
                    random_state=random_state
                )
        elif model_type == "XGBoost":
            if objective == 'classification':
                model = XGBClassifier(
                    objective='binary:logistic',
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state,
                    use_label_encoder=False,  # specific to XGBoost, to avoid deprecation warning
                    eval_metric="logloss"
                )
            elif objective == 'regression':  # regression
                model = XGBRegressor(
                    objective='reg:squarederror',
                    learning_rate=learning_rate,
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    random_state=random_state
                )
        else:
            print(f"Invalid model type '{model_type}'. Please choose 'lightGBM' or 'XGBoost'.")
            return None, None
        
        # Train the model
        model.fit(df_train[features], df_train[target].values.ravel())
        print(f"{model_type.capitalize()} Model Training Completed")
        
        # Predict the target on the test set
        predicted_target = model.predict(df_test[features])
    
        # Prepare a DataFrame for actual and predicted target values
        id_index = df_test.index.tolist()
        df_pred = pd.DataFrame({
            'task_id': id_index,
            'act_target': df_test[target].values.ravel(),
            'pred_target': predicted_target
        })
        # Apply a non-negative adjustment on predicted values (only if regression)
        if objective == 'regression':
            df_pred['adj_pred_target'] = df_pred['pred_target'].apply(lambda x: max(x, 0))
    
        # Print target min and max values for reference
        print("Target Range in Test Set:")
        print(f"Max value: {df_test[target].max()}")
        print(f"Min value: {df_test[target].min()}")
        
        max_target = df_test[target].max()
        max_rounded = round_up_to_one_sig_fig(max_target)
        min_target = df_test[target].min()
        min_rounded = round_down_to_one_sig_fig(min_target)
        
        
        # Generate a dummy plot for comparison
        if objective == 'regression':
            dm.dummy_plot(
                df_pred=df_pred, 
                target=target, 
                unit=unit, 
                path_to_save = path_to_save, 
                max_value = max_rounded, 
                min_value = min_rounded, 
                x_label = f"ML Prediction of {target} (${unit}$)", 
                y_label = f"DFT Calculation of {target} (${unit}$)", 
                adj=True
                )
        elif objective == 'classification':
            # Predict
            y_pred = model.predict_proba(df_test[features])
            y_pred_2 = model.predict(df_test[features])
            y_test = df_test[target]
            
            # Generate table of results
            
            # micro, macro, weighted
            strategy = 'weighted'
            
            print('1. The F-1 score of the model {}\n'.format(f1_score(y_test.ravel(), y_pred_2, average=strategy)))
            print('2. The recall score of the model {}\n'.format(recall_score(y_test.ravel(), y_pred_2, average=strategy)))
            print('3. Classification report \n {} \n'.format(classification_report(y_test.ravel(), y_pred_2, 
                                                                                target_names=['False', 'True'],
                                                                                digits=3)))
            print('4. Classification report \n {} \n'.format(multilabel_confusion_matrix(y_test.ravel(), y_pred_2)))
            print('5. Confusion matrix \n {} \n'.format(confusion_matrix(y_test.ravel(), y_pred_2)))
            print('6. Accuracy score \n {} \n'.format(accuracy_score(y_test.ravel(), y_pred_2)))
            print('7. Balanced accuracy score \n {} \n'.format(balanced_accuracy_score(y_test.ravel(), y_pred_2)))
            
                    # np.set_printoptions(precision=2)
            fontsize = 13
            fontsize2 = 10
            
            cm= confusion_matrix(y_test.ravel(), y_pred_2)
            
            if target == 'is_metal':
                disp = ConfusionMatrixDisplay(
                                            confusion_matrix = cm,
                                            display_labels=np.array(['Non-Metal', 'Metal'], dtype='<U10'),
                                            )
            elif target == 'is_stable':
                disp = ConfusionMatrixDisplay(
                                            confusion_matrix = cm,
                                            display_labels=np.array(['Unstable', 'Stable'], dtype='<U10'),
                                            )
            else:
                disp = ConfusionMatrixDisplay(
                                            confusion_matrix = cm,
                                            display_labels=np.array(['False', 'True'], dtype='<U10'),
                                            )
            disp.plot(cmap="viridis")
            # disp.ax_.set_title('(d)', fontsize=fontsize, y=1.01) 
            
            # plt.tick_params(axis='both', which='major', labelsize=fontsize, direction='in')
            # plt.xlabel('ML-based label', size=fontsize2)
            # plt.ylabel('DFT-based label', size=fontsize2)
            
            #Final_figure
            plt.savefig(path_to_save + target + 'confusion_matrix_dummy.png', dpi = 500, bbox_inches="tight")
            
            plt.show()
            
    
    
        # Calculate feature relevance scores and create a sorted DataFrame
        if model_type == "lightGBM":
            feature_importances = model.feature_importances_
        elif model_type == "XGBoost":
            feature_importances = model.feature_importances_
        
        feature_score = pd.DataFrame({'feature': features, 'relevance_score': feature_importances})
        feature_score = feature_score.sort_values(by='relevance_score', ascending=False).reset_index(drop=True)
    
        # Plot the top features based on relevance score
        sns.set(rc={'figure.figsize': (10, 10)})
        plot = sns.barplot(x='relevance_score', y='feature', data=feature_score[:no_of_features_to_plot])
        plot.set(xlabel='Relevance score', ylabel='Feature')
        print(f"Top {no_of_features_to_plot} Features by Relevance Score Plot Generated")    
   
    return df_pred, feature_score