#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from sklearn.metrics import mean_absolute_error, r2_score, max_error
from sklearn.metrics import explained_variance_score,mean_squared_error


def dummy_plot(df_pred, target, unit, path_to_save, max_value, min_value, x_label, y_label, adj=True):
        
        # Define variables
        y_test = df_pred['act_target']
        y_pred_adj = df_pred['adj_pred_target']
        y_pred = df_pred['pred_target']
        df_pred[df_pred['act_target']<0.2]    
    
        X = df_pred['act_target']
        if adj is True:
                Y = df_pred['adj_pred_target']
        else:
                Y = df_pred['pred_target']

        # Stats
        results = sm.OLS(Y,sm.add_constant(X)).fit()
        print(results.summary())

        # Figure
        fig = plt.figure(figsize=(10, 10)) #, dpi=100, facecolor='w', edgecolor='k')
        gs = GridSpec(4, 4)

        ax_scatter = fig.add_subplot(gs[1:4, 0:3])
        ax_hist_y = fig.add_subplot(gs[0,0:3])
        ax_hist_x = fig.add_subplot(gs[1:4, 3])

        fontsize = 20

        # Predicted vs Actual
        ax_scatter.plot(X, Y, 'o', markersize=6, color='black', alpha=0.1)

        offset = 0.15

        # line of best fit
        linear_fit = np.linspace(min_value-offset, max_value-offset, 10)
        ax_scatter.plot(linear_fit, linear_fit*results.params[1] + results.params[0], linewidth=2, color='blue', alpha=1.0, label='Line of Best Fit') # '-.',

        # Ideal y=x 
        y = x = np.linspace(min_value-offset, max_value-offset, 10)
        ax_scatter.plot(x, y, '--', linewidth=2, color='red', alpha=1.0, label='Ideal Regressor')
        ax_scatter.legend(loc='upper left',fontsize=fontsize)
                
        ticks = np.linspace(min_value, max_value, 6)

        onset = 0.5
        ax_scatter.set_xlabel(x_label, fontsize=fontsize)
        ax_scatter.set_ylabel(y_label, fontsize=fontsize)
        ax_scatter.tick_params(axis='both', which='both', labelsize=fontsize, direction="in")
        ax_scatter.set_xlim(min_value-onset, max_value+onset)
        ax_scatter.set_ylim(min_value-onset, max_value+onset)
        ax_scatter.set_xticks(ticks)
        ax_scatter.set_yticks(ticks)

        ax_hist_x.tick_params(axis='both', which='both', labelsize=fontsize, direction="in", labelleft=False)
        ax_hist_x.set_xlabel('Count', fontsize=fontsize)
        ax_hist_x.set_ylim(min_value-onset, max_value+onset)
        ax_hist_x.set_xticks([0, 20, 40, 60, 80, 100])
        ax_hist_x.set_yticks(ticks)

        ax_hist_y.tick_params(axis='both', which='both', labelsize=fontsize, direction="in", labelbottom=False)
        ax_hist_y.set_ylabel('Count', fontsize=fontsize)
        ax_hist_y.set_xlim(min_value-onset, max_value+onset)
        ax_hist_y.set_yticks([0, 20, 40, 60, 80, 100])
        ax_hist_y.set_xticks(ticks)

        # Distribution plots
        alpha = 1.0
        histtype = 'bar' #'step' bar
        color = 'skyblue' #'black' #'tab:grey'
        bins = 50 #70

        ax_hist_y.hist(
                        y_test, 
                        bins=bins, 
                        color=color, 
                        histtype=histtype,
                        alpha=alpha
                        )
        ax_hist_x.hist(
                        y_pred_adj, 
                        orientation='horizontal', 
                        bins=bins, 
                        color=color, 
                        histtype=histtype,
                        alpha=alpha
                        )


        #Text in figure
        font1 = {'family': 'Arial',
                'color':  'red',
                'weight': 'normal',
                'size': fontsize-3,
                }

        font2 = {'family': 'Arial',
                'color':  'blue',
                'weight': 'normal',
                'size': fontsize-3,
                }

        font3 = {'family': 'Arial', #'serif',
                'color':  'black',
                'weight': 'normal',
                'size': fontsize-3,
                }

        ax_scatter.text(max_value*0.42, max_value*0.55, r'$y = x$', fontdict=font1)
        ax_scatter.text(max_value*0.55, max_value*0.45, r'$y = ' + str(round(results.params[1],3)) + 'x - ' + str(round(results.params[0],3)) + '$', fontdict=font2)

        xx = max_value*0.6
        r2 = r2_score(X, Y)
        ax_scatter.text(xx, 1.55, r'$R^{2}$ = ' + str(round(r2,3)), fontdict=font3)

        mae = mean_absolute_error(X, Y)
        ax_scatter.text(xx, 0.8, r'$MAE$ = ' + str(round(mae,3)) + ' ' +  str(unit) , fontdict=font3)

        mse = mean_squared_error(X, Y)
        rmse = np.sqrt(mse)
        ax_scatter.text(xx, 0., r'$RMSE$ = ' + str(round(rmse,3)) + ' ' + str(unit), fontdict=font3)

        # plt.gca().spines['top'].set_linewidth(3)
        # plt.gca().spines['bottom'].set_linewidth(3)
        # plt.gca().spines['left'].set_linewidth(3)
        # plt.gca().spines['right'].set_linewidth(3)
        # plt.gca().spines['top'].set_color('black')
        # plt.gca().spines['bottom'].set_color('black')
        # plt.gca().spines['left'].set_color('black')
        # plt.gca().spines['right'].set_color('black')
        
        # plt.gca().xaxis.set_ticks_position('both')
        # plt.gca().yaxis.set_ticks_position('both')

        # plt.gca().tick_params(axis='both', which='major', direction='in', width=3, length=10, color='black')
        # plt.gca().tick_params(axis='both', which='minor', direction='in', width=3, length=5, color='black')



        #final_figure
        fig.savefig(path_to_save + 'regression_' + target + '.png', dpi = 500, bbox_inches="tight")
        print('Saved:', 'regression_' + target + '.png')


        # Save the regression summary and metrics to a file and print them to the terminal
        summary_file_path = f"{path_to_save}/{target}_summary"
        
        with open(f"{summary_file_path}.txt", "w") as f:
            # Write and print the OLS Regression summary
            summary_text = results.summary().as_text()
            f.write(summary_text + "\n\n")
            
            # Extract and print model parameters
            m = results.params[1]
            c = results.params[0]
            f.write(f"m = {m}\n")
            f.write(f"c = {c}\n\n")
            print(f"m = {m}")
            print(f"c = {c}\n")
        
            # Calculate and print additional metrics
            mae = mean_absolute_error(X, Y)
            mse = mean_squared_error(X, Y)
            # rmse = mean_squared_error(X, Y, squared=False)
            rmse = np.sqrt(mse)
            r_squared = r2_score(X, Y)
            max_err = max_error(X, Y)
            explained_var = explained_variance_score(X, Y, multioutput='variance_weighted')
        
            # Log to file and print
            metrics = (
                f"MAE: {mae}\n"
                f"MSE: {mse}\n"
                f"RMSE: {rmse}\n"
                f"R-squared: {r_squared}\n"
                f"Max error: {max_err}\n"
                f"Explained Variance Score: {explained_var}\n"
            )
            f.write(metrics)
            print(metrics)
       
        print(f"Summary Data saved to {summary_file_path}.txt") 
       
        # Save the predicted values and ground truth to a csv file and print them to the terminal
        prediction_file_path = f"{path_to_save}/{target}_prediction" 
        df_pred.to_csv(f"{prediction_file_path}.csv")
        print(df_pred.head())
        print(f"Prediction Data saved to {prediction_file_path}.csv")

        # print('m = ', results.params[1])
        # print('c = ', results.params[0], '\n')

        # print('MAE: ', mean_absolute_error(X, Y))
        # print('MSE: ', mean_squared_error(X, Y))
        # print('RMSE: ', mean_squared_error(X, Y, squared=False))
        # print('R-squared: ', r2_score(X, Y))
        # print('Max error: ', max_error(X, Y))
        # print('Explained_variance_score: ', explained_variance_score(X, Y, multioutput='variance_weighted'))

##########################################################################################################################