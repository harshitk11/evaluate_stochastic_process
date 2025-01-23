"""
For s-level 80 (sameInit 1000 simulations) 
- Load the file containing the varzt scores
- Load the raw prediction, GT 

- Design variable: num_sim
    - bunch up predictions for num_sim simulations -> calculate the evaluation metric -> calculate the variance over all the bunchups
    
- Plot the metric SD vs varzt

"""
from predictive_difficulty_score_analysis import load_SD_vs_time_step_dict
from reliability_analysis import load_raw_pred_forecastGT_dict, raw_pred_forecastGT_dict
from ece_counterexample_study import get_metrics, calculate_correlation, calculate_energy_score

import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.cm import viridis
import seaborn as sns


import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class metric_vs_time_combined_simulation:
    # Method that generates metrics for all simulations combined and then calculates the SD across groups of samples
    def generate_metrics_combined_simulation(stochasticity_list, dnn_name, dataset_type, num_samples=1000, bunchup_samples=5, ece_bins=15):
        slevel_metric_timestep = {}
       
        metric_names = ['ece', 'precision', 'recall', 'auc_pr', 'mse', 'bce', 'crps', 'correlation', 'energy_score', 'f1_score', 'auc_roc']
        
        for stochasticity_level in stochasticity_list:
            print(f"Stochasticity Level: {stochasticity_level}")
            slevel_metric_timestep[stochasticity_level] = {}
            # Initialize metric dictionaries
            for metric_name in metric_names:
                slevel_metric_timestep[stochasticity_level][metric_name] = {}
            
            # Load prediction and ground truth data
            raw_pred_forecastGT_dict = load_raw_pred_forecastGT_dict(dnn_name, stochasticity_level, dataset_type=dataset_type)
            
            for timestep in range(10, 59):
                print(f"Time Step: {timestep}")
                
                # Initialize a dictionary to store metrics for each group
                metrics_per_group = {metric_name: [] for metric_name in metric_names}
                
                num_groups = num_samples // bunchup_samples
                for group_idx in range(num_groups):
                    y_prob = []
                    y_true = []
                    
                    for sample_idx in range(group_idx * bunchup_samples, (group_idx + 1) * bunchup_samples):
                        try:
                            test_sample = raw_pred_forecastGT_dict[sample_idx]
                            prediction = test_sample['prediction']
                            observedGT = test_sample['observedGT']
                            
                            prediction_timestep = prediction[timestep]
                            observedGT_timestep = observedGT[timestep]
                            
                            y_prob.append(prediction_timestep.flatten())
                            y_true.append(observedGT_timestep.flatten())
                        
                        except: # Certain simulations end early so we skip them
                            print(f"Error in sample {sample_idx}")
                            print(len(prediction), len(observedGT))
                            continue
                    
                    if len(y_prob) == 0 or len(y_true) == 0:
                        continue
                    
                    y_prob_np = np.concatenate(y_prob).flatten()
                    y_true_np = np.concatenate(y_true).astype(int).flatten()
                        
                    # Calculate the evaluation metrics for the current group
                    ece, precision, recall, auc_pr, mse, bce, crps, f1_score, auc_roc = get_metrics(y_true_np, y_prob_np, ece_bins=ece_bins)
                    correlation_val = calculate_correlation(y_prob_np, y_true_np)
                    energy_score_val = calculate_energy_score(y_true_np, y_prob_np)
                    
                    # Store the metric values for this group
                    metrics_per_group['ece'].append(ece)
                    metrics_per_group['precision'].append(precision)
                    metrics_per_group['recall'].append(recall)
                    metrics_per_group['auc_pr'].append(auc_pr)
                    metrics_per_group['mse'].append(mse)
                    metrics_per_group['bce'].append(bce)
                    metrics_per_group['crps'].append(crps)
                    metrics_per_group['correlation'].append(correlation_val)
                    metrics_per_group['energy_score'].append(energy_score_val)
                    metrics_per_group['f1_score'].append(f1_score)
                    metrics_per_group['auc_roc'].append(auc_roc)
                
                # Compute the standard deviation of the metrics across groups
                for metric_name in metric_names:
                    metric_values = metrics_per_group[metric_name]
                    if len(metric_values) > 0:
                        SD_value = np.std(metric_values)
                        slevel_metric_timestep[stochasticity_level][metric_name][timestep] = SD_value
                    else:
                        slevel_metric_timestep[stochasticity_level][metric_name][timestep] = np.nan  # Assign NaN if no data
    
        return slevel_metric_timestep
    
    # Function to plot SD vs Time for the metrics
    def plot_metric_SD_vs_time(slevel_metric_timestep, stochasticity_level, dnn_name, savepath_plots, varzt_scores, normalize=False):
        os.makedirs(savepath_plots, exist_ok=True)
        
        # Extract data for the given stochasticity level
        metrics = slevel_metric_timestep[stochasticity_level]
        timesteps = sorted(next(iter(metrics.values())).keys())  # Assuming all metrics have the same timesteps

        # Set a professional color palette
        colors = plt.get_cmap('tab10')
        
        # Prepare the plot with a professional style
        plt.style.use('classic')
        plt.figure(figsize=(8, 5))
        
        # Define markers and create an iterator
        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', '+', 'x', 'd', '|', '_']
        marker_iter = iter(markers)
        
        # Plot each metric with a different color and marker
        for i, metric_name in enumerate(metrics.keys()):
            if metric_name in ['mse', 'auc_pr', 'precision', 'recall', 'ece']:
                SD_values = [metrics[metric_name][timestep] for timestep in timesteps]
                varzt_values = [varzt_scores[timestep] for timestep in timesteps]
                
                # Normalize the standard deviation if the flag is True
                if normalize:
                    max_SD = np.nanmax(SD_values)
                    SD_values = [value / max_SD if max_SD != 0 else 0 for value in SD_values]
                
                # Map metric names for better readability
                metric_label = {
                    "auc_pr": "AUC-PR",
                    "auc_roc": "AUC-ROC",
                    "crps": "CRPS",
                    "bce": "BCE",
                    "mse": "MSE",
                    "ece": "ECE",
                    "f1_score": "F1 Score",
                    "energy_score": "Energy Score",
                    "correlation": "Correlation",
                    "precision": "Precision",
                    "recall": "Recall"
                }.get(metric_name, metric_name.capitalize())
                
                # Add a line for this metric to the plot with different markers
                plt.plot(varzt_values, SD_values, marker=next(marker_iter), linestyle='--', color=colors(i),
                        label=metric_label, markersize=8, zorder=3)
        
        # Configure plot aesthetics
        plt.xlabel(r'$\mathit{Var}(Z_t)$', fontsize=18, fontweight='bold')
        plt.ylabel('Normalized Standard Deviation' if normalize else 'Standard Deviation of Metric', fontsize=18, fontweight='bold')
        
        # Set grid with transparency (alpha) and behind plot elements (zorder)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.2, zorder=1)
        
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        
        legend = plt.legend(
        title='Metric',
        fontsize=10,
        title_fontsize=12,
        ncol=5,
        loc='upper center',  # Position relative to the Axes
        bbox_to_anchor=(0.5, 1.15)  # Place it at the top center, slightly above the plot
        )
        plt.setp(legend.get_texts(), fontsize=10)
        
        # Save the plot
        plot_filename = os.path.join(savepath_plots, f'{"normalized_" if normalize else ""}SD_vs_time_S-Level_{stochasticity_level}_{dnn_name}.pdf')
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        plt.close()



def main():
    stochasticity_list = [80]
    
    base_savepath = "/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/metric_vs_varzt"
    os.makedirs(base_savepath, exist_ok=True)
    
    SD_vs_time_step_dict = load_SD_vs_time_step_dict()
    SD_vs_time_step_dict_slevel80 = SD_vs_time_step_dict[80]
    # for k, v in SD_vs_time_step_dict_slevel80.items():
    #     print(k, v)
    # exit()
        
    # Assuming you have the necessary functions and data loaded:
    stochasticity_list = [80]
    dnn_name = "d-convlstm"
    dataset_type = 'sameInit'
    num_samples = 1000
    bunchup_samples = 50
    ece_bins = 15

    # Generate metrics and compute SD
    slevel_metric_timestep = metric_vs_time_combined_simulation.generate_metrics_combined_simulation(
        stochasticity_list,
        dnn_name,
        dataset_type,
        num_samples,
        bunchup_samples,
        ece_bins
    )

    # Plot SD vs Time for a specific stochasticity level
    savepath_plots = os.path.join(base_savepath, f'plots_{num_samples}_{bunchup_samples}')
    metric_vs_time_combined_simulation.plot_metric_SD_vs_time(
        slevel_metric_timestep,
        stochasticity_level=stochasticity_list[0],
        dnn_name=dnn_name,
        savepath_plots=savepath_plots,
        normalize=False,
        varzt_scores=SD_vs_time_step_dict_slevel80
    )

       
if __name__ == "__main__":
    main()