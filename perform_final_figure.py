#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import math
import joblib
import seaborn as sns
import contextlib

import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.preprocessing import label_binarize
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import jaccard_score, multilabel_confusion_matrix, roc_curve, roc_auc_score, \
                            auc, f1_score, classification_report, recall_score, precision_recall_curve, \
                            balanced_accuracy_score, confusion_matrix, accuracy_score, average_precision_score, \
                            hamming_loss, matthews_corrcoef
from sklearn.manifold import TSNE

from matplotlib.ticker import MultipleLocator, AutoMinorLocator

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

def perform_final_figure(
    file_path, file_name, target, objective, unit, 
    boosting_type='gbdt', importance_type='gain', 
    scaled=False, no_of_features_to_plot=30, 
    max_depth=-1, random_state=42
):
    """
    Loads optimized model parameters and performs model evaluation and feature importance ranking.
    
    Parameters:
    - file_path (str): Path to the directory where files are stored.
    - file_name (str): Identifier for the dataset.
    - target (str): Target variable for analysis.
    - objective (str): 'regression' or 'classification'.
    - unit (str): Unit of Target Variable.
    - boosting_type (str): Type of boosting algorithm (e.g., 'gbdt', 'dart', 'goss'). Default is 'gbdt'.
    - importance_type (str): Type of feature importance measure ('gain', 'split'). Default is 'gain'.
    - scaled (bool): Whether the data is scaled; required for classification. (Default is False).
    - no_of_features_to_plot (int): Number of features to plot in the feature relevance plot. (Default is 30)
    - max_depth (int): Maximum depth of the individual estimators. (Default is -1, no limit).
    - random_state (int): Random seed for reproducibility. (Default is 42).
    
    Returns:
    - feature_score (pd.DataFrame): DataFrame of features ranked by relevance score.
    """
    
    print("\n******************\nperform_final_figure Function Running\n******************\n")
    
    # Define paths
    path_to_save = f"{file_path}/pkl/{target}_results"
    path_to_save_final = f"{path_to_save}/final/"
    
    # Create results directory if it doesn't exist
    os.makedirs(path_to_save_final, exist_ok=True)
    
    log_path = f"{path_to_save}/log/output_log_09_{__name__}.txt"
    with open(log_path, 'w') as f, contextlib.redirect_stdout(f):
    
        path_to_train_file = f"{path_to_save}/df_train_{target}_engineered.pkl"
        path_to_test_file = f"{path_to_save}/df_test_{target}_engineered.pkl"
        path_to_features = f"{path_to_save}/features_selected_from_RFE_{target}.pkl"
        
        # Load features and data
        features = joblib.load(path_to_features)
        print(f"Number of Features selected by Recursive Feature Elimination: {len(features)}")
    
        df_data = joblib.load(f"{file_path}/pkl/{file_name}_data.pkl")
    
        
        if objective == 'regression':
            max_target = df_data[target].max()
            max_rounded = round_up_to_one_sig_fig(max_target)
            min_target = df_data[target].min()
            min_rounded = round_down_to_one_sig_fig(min_target)
    
        # Print target stats
        print(f"Target Variable: {target}")
        if objective == 'regression':
            print('Max value: ', max_target)
            print('Max value rounded: ', max_rounded)
            print('Min value: ', min_target)
            print('Min value rounded: ', min_rounded)
        print(f"Objective: {objective}")
        
        df_train = joblib.load(path_to_train_file)
        df_test = joblib.load(path_to_test_file)
        
        # Load optimized hyperparameters
        bayesian_opt = joblib.load(f"{path_to_save}/optimization_data_{target}.pkl")
        learning_rate = float(bayesian_opt['learning_rate'].iloc[0])
        n_estimators = int(bayesian_opt['n_estimators'].iloc[0])
        num_leaves = int(bayesian_opt['num_leaves'].iloc[0])
        
        print(f"Learning Rate from Bayesian Optimization: {learning_rate}")
        print(f"Number of Estimators from Bayesian Optimization: {n_estimators}")
        print(f"Number of Leaves from Bayesian Optimization: {num_leaves}")
        
        # Choose model type based on objective
        if objective == 'regression':
            model = LGBMRegressor(
                boosting_type=boosting_type,
                objective='regression',
                importance_type=importance_type,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                num_leaves=num_leaves,
                max_depth=max_depth,
                random_state=random_state,
                verbose=-1
            )
        else:
            model = LGBMClassifier(
                boosting_type=boosting_type,
                objective='binary',
                importance_type=importance_type,
                learning_rate=learning_rate,
                n_estimators=n_estimators,
                num_leaves=num_leaves,
                max_depth=max_depth,
                random_state=random_state,
                verbose=-1
            )
    
        # Train model
        model.fit(df_train[features], df_train[target].values.ravel())
        print(model)
        
        # Write model to file
        joblib.dump(model, f"{path_to_save_final}/{file_name}_{target}_final_model.pkl") 
        
        # Predict the target value
        predicted_target = model.predict(df_test[features])
        
        # Prepare result DataFrame
        df_pred = pd.DataFrame({
            'task_id': df_test.index,
            'act_target': df_test[target].values.ravel(),
            'pred_target': predicted_target
        })
        df_pred['adj_pred_target'] = df_pred['pred_target'].apply(lambda x: max(0, x))
        
        # Define prediction variables
        y_test = df_pred['act_target']
        # y_pred_adj = df_pred['adj_pred_target']
        
        # Generate a dummy plot for comparison
        if objective == 'regression':
            dm.dummy_plot(
                df_pred=df_pred, 
                target=target, 
                unit=unit, 
                path_to_save = path_to_save_final, 
                max_value = max_rounded, 
                min_value = min_rounded, 
                x_label = f"GBFS Prediction of Bandgap [${unit}$]", # "GBFS Prediction of {target} (${unit}$)", 
                y_label = f"DFT Calculation of Bandgap [${unit}$]", # "DFT Prediction of {target} (${unit}$)",
                adj=True
                )
            
            # Generate residual plot
            # Calculate residual error
            df_pred['residual_error'] = df_pred['act_target'] - df_pred['adj_pred_target']
            
            residual_summary = df_pred['residual_error'].describe()
            print("\nResidual Error Summary Statistics:")
            print(residual_summary)
            residual_mean = round(residual_summary['mean'], 3)
            residual_std = round(residual_summary['std'], 3)
            
            # Plot Histogram
            plt.figure(figsize=(10,10))
            sns.set(font_scale=2.5)
            sns.set_style("ticks")
            bins = np.arange(-2.5,2.6,0.1)
            plt.hist(df_pred['residual_error'], bins=bins, alpha=1.0, color='skyblue')
            plt.xlabel('Residual Error [eV]')
            plt.ylabel('Frequency')
            plt.xlim([-2.5,2.5])
            plt.text(0.2, 200, 'Residual Error:')
            plt.text(0.2, 185, f'Mean = {residual_mean}')
            plt.text(0.2, 170, f'Std. Dev. = {residual_std}')
            
            plt.gca().spines['top'].set_linewidth(3)
            plt.gca().spines['bottom'].set_linewidth(3)
            plt.gca().spines['left'].set_linewidth(3)
            plt.gca().spines['right'].set_linewidth(3)
            plt.gca().spines['top'].set_color('black')
            plt.gca().spines['bottom'].set_color('black')
            plt.gca().spines['left'].set_color('black')
            plt.gca().spines['right'].set_color('black')
            
            plt.gca().xaxis.set_ticks_position('both')
            plt.gca().yaxis.set_ticks_position('both')
    
            plt.gca().tick_params(axis='both', which='major', direction='in', width=3, length=10, color='black')
            plt.gca().tick_params(axis='both', which='minor', direction='in', width=3, length=5, color='black')
                  
            plt.savefig(f"{path_to_save_final}{file_name}_{target}_residual_histogram.png", dpi = 500, bbox_inches="tight")
            plt.show()
            
        elif objective == 'classification':
            # Predict
            y_pred = model.predict_proba(df_test[features])
            y_pred_2 = model.predict(df_test[features])
            y_test = df_test[target]
            
            if target == 'is_metal':
                target_class_names = ['Non-Metal', 'Metal']
                target_class_title = 'Confusion Matrix for 2D Material Metallicity'
                target_class_xlabel = 'GBFS Predicted Metallicity'
                target_class_ylabel = 'DFT Predicted Metallicity'
            elif target == 'is_stable':
                target_class_names = ['Unstable', 'Stable']
                target_class_title = 'Confusion Matrix for 2D Material Stability'
                target_class_xlabel = 'GBFS Predicted Stability'
                target_class_ylabel = 'DFT Predicted Stability'
            elif target == 'is_gap_direct':
                target_class_names = ['Indirect', 'Direct']
                target_class_title = 'Confusion Matrix for 2D Material Bandgap Type'
                target_class_xlabel = 'GBFS Predicted Bandgap Type'
                target_class_ylabel = 'DFT Predicted Bandgap Type'        
            else:
                target_class_names = ['False', 'True']
                target_class_title = 'Confusion Matrix for Material Property'
                target_class_xlabel = 'GBFS Predicted Property'
                target_class_ylabel = 'DFT Predicted Property'
            
            # Generate and save table of results
            
            # micro, macro, weighted
            strategy = 'weighted'
            
            print('1. The F-1 score of the model {}\n'.format(f1_score(y_test.ravel(), y_pred_2, average=strategy)))
            print('2. The recall score of the model {}\n'.format(recall_score(y_test.ravel(), y_pred_2, average=strategy)))
            print('3. Classification report \n {} \n'.format(classification_report(y_test.ravel(), y_pred_2, 
                                                                                target_names=target_class_names,
                                                                                digits=3)))
            print('4. Classification report \n {} \n'.format(multilabel_confusion_matrix(y_test.ravel(), y_pred_2)))
            print('5. Confusion matrix \n {} \n'.format(confusion_matrix(y_test.ravel(), y_pred_2)))
            print('6. Accuracy score \n {} \n'.format(accuracy_score(y_test.ravel(), y_pred_2)))
            print('7. Balanced accuracy score \n {} \n'.format(balanced_accuracy_score(y_test.ravel(), y_pred_2)))
            print('8. AUC-ROC Score \n {} \n'.format(roc_auc_score(y_test.ravel(), y_pred_2)))
    
            with open(f"{path_to_save_final}{file_name}_{target}_model_evaluation_results.txt", 'w') as file:
                file.write('1. The F-1 score of the model {}\n'.format(f1_score(y_test.ravel(), y_pred_2, average=strategy)))
                file.write('2. The recall score of the model {}\n'.format(recall_score(y_test.ravel(), y_pred_2, average=strategy)))
                file.write('3. Classification report \n{} \n'.format(classification_report(y_test.ravel(), y_pred_2, 
                                                                                          target_names=target_class_names,
                                                                                          digits=3)))
                file.write('4. Multilabel confusion matrix \n{} \n'.format(multilabel_confusion_matrix(y_test.ravel(), y_pred_2)))
                file.write('5. Confusion matrix \n{} \n'.format(confusion_matrix(y_test.ravel(), y_pred_2)))
                file.write('6. Accuracy score \n{} \n'.format(accuracy_score(y_test.ravel(), y_pred_2)))
                file.write('7. Balanced accuracy score \n{} \n'.format(balanced_accuracy_score(y_test.ravel(), y_pred_2)))
                file.write('8. AUC-ROC Score \n{} \n'.format(roc_auc_score(y_test.ravel(), y_pred_2)))
            
            # np.set_printoptions(precision=2)
            fontsize = 13
            fontsize2 = 10
                   
            cm = confusion_matrix(y_test.ravel(), y_pred_2)
            labels = np.unique(y_test.ravel())
            
            # Plotting
            plt.figure(figsize=(10, 10))
            sns.set(font_scale=2.5)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                        xticklabels=target_class_names, yticklabels=target_class_names)
            plt.xlabel(target_class_xlabel)
            plt.ylabel(target_class_ylabel)
            plt.title(target_class_title)
            plt.tight_layout()
    
            # disp.ax_.set_title('(d)', fontsize=fontsize, y=1.01) 
            
            # plt.tick_params(axis='both', which='major', labelsize=fontsize, direction='in')
            # plt.xlabel('ML-based label', size=fontsize2)
            # plt.ylabel('DFT-based label', size=fontsize2)
            
            #Final Confusion Matrix Figure
            plt.savefig(f"{path_to_save_final}{file_name}_{target}_confusion_matrix_final.png", dpi = 500, bbox_inches="tight")
            plt.show()
            
            fpr, tpr, thresholds_roc = roc_curve(y_test, y_pred[:, 1])
            df_roc = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Thresholds':thresholds_roc})
            joblib.dump(df_roc, f"{path_to_save_final}/{file_name}_{target}_final_roc.pkl")
            
            precision, recall, thresholds_pr = precision_recall_curve(y_test, y_pred[:, 1])
            df_pr = pd.DataFrame({'Precision': precision, 'Recall': recall})
            joblib.dump(df_pr, f"{path_to_save_final}/{file_name}_{target}_final_pr.pkl")
            
            
            
            # ### t-SNE Visual ###
            
            # # Use t-SNE for dimensionality reduction
            # # Option 1: Use raw features 
            # x_embedded = TSNE(n_components=2, perplexity=20, max_iter=1000, learning_rate=10.0, random_state=42).fit_transform(df_test)
            
            # # Option 2: Use model's leaf embeddings
            # # x_embedded = TSNE(n_components=2, random_state=42).fit_transform(x_leaves)        
            
            # # Plot the t-SNE results
            # plt.figure(figsize=(8,6))
            # scatter = plt.scatter(x_embedded[:,0], x_embedded[:,1], c=y_test, cmap='viridis', alpha=0.7)
            # plt.colorbar(scatter, label='Classes')
            # plt.show()
            
        # Feature importance ranking
        feature_score = pd.DataFrame({
            'feature': features,
            'relevance_score': model.feature_importances_
        }).sort_values(by='relevance_score', ascending=False).reset_index(drop=True)
        
        # Plot feature relevance scores
        sns.set(rc={'figure.figsize': (10, 10)})
        sns.set(font_scale=2.5)
        sns.set_style("ticks")
        fig = sns.barplot(x='relevance_score', y='feature', data=feature_score[:no_of_features_to_plot],color='skyblue')
        
        # for tick in fig.get_yticklabels():
        #     tick.set_position((0.05, tick.get_position()[1]))
        #     tick.set_horizontalalignment('left')
        
        # fig.tick_params(axis='y', direction='in', length=6)
        
        
        # for p in fig.patches:
        #     patch_label = p.get_yticklabels() # int(p.get_width())
        #     fig.annotate(
        #             patch_label,
        #             (p.get_width() * 0.5, p.get_y() + p.get_height() / 2),
        #             ha='left',
        #             va='center',
        #             color='black',
        #             fontsize=25
        #             )
        
        for tick in fig.get_yticklabels():
            label_text = f"{tick.get_text()}"
            fig.annotate(
                    label_text,
                    (50, tick.get_position()[1]),
                    ha='left',
                    va='center',
                    color='black',
                    fontsize=20
                    )
        fig.set_yticklabels([])
        fig.set(xlabel='Feature Relevance Score', ylabel='Compositional Feature Name')
        
        fig.spines['top'].set_linewidth(3)
        fig.spines['bottom'].set_linewidth(3)
        fig.spines['left'].set_linewidth(3)
        fig.spines['right'].set_linewidth(3)
        fig.spines['top'].set_color('black')
        fig.spines['bottom'].set_color('black')
        fig.spines['left'].set_color('black')
        fig.spines['right'].set_color('black')
        
        fig.xaxis.set_ticks_position('both')
    
        fig.tick_params(axis='both', which='major', direction='in', width=3, length=10, color='black')
        # fig.tick_params(axis='both', which='minor', direction='in', width=3, length=5, color='black')
    
        fig.xaxis.set_major_locator(MultipleLocator(2000))
        # fig.xaxis.set_minor_locator(MultipleLocator(200))
        
        plt.savefig(f"{path_to_save_final}{file_name}_{target}_relevance_score_final.png", dpi = 500, bbox_inches="tight")
        plt.show()
        
        
        joblib.dump(feature_score,f"{path_to_save_final}/{file_name}_{target}_final_features.pkl")
        joblib.dump(df_pred,f"{path_to_save_final}/{file_name}_{target}_pred.pkl")
    
    return feature_score, df_pred