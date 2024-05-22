"""
Code used to generate the plot showing the long horizon predictions of convLSTM-CA using the different evaluation metrics
(Figure 6 and Figure 14 of the paper)
"""

from reliability_analysis import load_raw_pred_forecastGT_dict
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os 
from sklearn.calibration import calibration_curve
import calibration as cal
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc, mean_squared_error
import pickle

# Method that takes as input y_true and y_prob and returns the ECE, Recall, Precision, AUC-PR and MSE
def get_metrics(y_true, y_prob):
    ece = cal.get_ece(y_prob, y_true)
    precision_val = precision_score(y_true, y_prob > 0.5)
    recall_val = recall_score(y_true, y_prob > 0.5)
    mse_val = mean_squared_error(y_true, y_prob)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
    auc_pr_val = auc(recall_curve, precision_curve)
    return ece, precision_val, recall_val, auc_pr_val, mse_val


class metric_vs_time_combined_simulation:
    # Method that generates metrics for all simulations combined and then calculates the mean and variance across simulations
    def generate_metrics_combined_simulation(stochasticity_list, dnn_name, base_savepath, dataset_type, load_pickle, num_samples=100, forecast_GT_folder=None):
        slevel_metric_timestep = {}
        
        if load_pickle:
            # Check if pickle file already exist, if yes, load the pickle file
            if os.path.exists(os.path.join(base_savepath, "combined_simulation","pickle_file",dnn_name, "slevel_metric_timestep.pkl")):
                with open(os.path.join(base_savepath, "combined_simulation","pickle_file",dnn_name, "slevel_metric_timestep.pkl"), "rb") as f:
                    slevel_metric_timestep = pickle.load(f)
                return slevel_metric_timestep
        
        
        for i, stochasticity_level in enumerate(stochasticity_list):
            print(f"Stochasticity Level: {stochasticity_level}")
            slevel_metric_timestep[stochasticity_level] = {}
            # Add entries for metric
            slevel_metric_timestep[stochasticity_level]['ece'] = {}
            slevel_metric_timestep[stochasticity_level]['precision'] = {}
            slevel_metric_timestep[stochasticity_level]['recall'] = {}
            slevel_metric_timestep[stochasticity_level]['auc_pr'] = {}
            slevel_metric_timestep[stochasticity_level]['mse'] = {}
            
            # print(f"Loading raw_pred_forecastGT_dict for s-level: {stochasticity_level}| dnn_name: {dnn_name}")
            raw_pred_forecastGT_dict = load_raw_pred_forecastGT_dict(dnn_name, stochasticity_level, dataset_type=dataset_type,
                                                                     base_loadpath=forecast_GT_folder)
            
            
            for timestep in range(10,59):
                print(f"Time Step: {timestep}")
                
                y_prob = []
                y_true = []
                
                for j in range(num_samples):
                    # print(f"Sample: {j}")
                    try:
                        test_sample = raw_pred_forecastGT_dict[j]
                        prediction = test_sample['prediction']
                        observedGT = test_sample['observedGT']
                        # print(f"Prediction Shape: {prediction.shape}")
                        # print(f"ObservedGT Shape: {observedGT.shape}")
                        
                        prediction_timestep = prediction[timestep]
                        observedGT_timestep = observedGT[timestep]
                        
                        y_prob.append(prediction_timestep.flatten())
                        y_true.append(observedGT_timestep.flatten())
                    
                    except:
                        continue
                    
                y_prob_np = np.concatenate(y_prob)
                y_true_np = np.concatenate(y_true).astype(int)
                    
                # Calculate and store the evaluation metrics for the current timestep
                ece, precision, recall, auc_pr, mse = get_metrics(y_true_np.flatten(), y_prob_np.flatten())
                
                slevel_metric_timestep[stochasticity_level]['ece'][timestep] = ece
                slevel_metric_timestep[stochasticity_level]['precision'][timestep] = precision
                slevel_metric_timestep[stochasticity_level]['recall'][timestep] = recall
                slevel_metric_timestep[stochasticity_level]['auc_pr'][timestep] = auc_pr
                slevel_metric_timestep[stochasticity_level]['mse'][timestep] = mse
                
        # Save the sleve_metric_timestep dictionary to a pickle file
        base_savepath = os.path.join(base_savepath, "combined_simulation","pickle_file",dnn_name)
        os.makedirs(base_savepath, exist_ok=True)
        with open(os.path.join(base_savepath, "slevel_metric_timestep.pkl"), "wb") as f:
            pickle.dump(slevel_metric_timestep, f)
        
        # Generate the plots
        metric_vs_time_combined_simulation.plot_metrics_for_combined_simulation(slevel_metric_timestep, base_savepath)
        
        return slevel_metric_timestep
    
    def plot_metrics_for_combined_simulation(slevel_metric_timestep, savepath):
        metrics = ['ece', 'precision', 'recall', 'auc_pr', 'mse']
        colors = plt.get_cmap('tab10').colors  # Use tab10 colormap for a professional look

        # Define marker styles and create an iterator for them, as per the previous style
        marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd', '|', '_']
        
        for metric in metrics:
            plt.figure(figsize=(8, 6))  # Adjusted to match the previous style's figure size

            for i, (stochasticity_level, data) in enumerate(slevel_metric_timestep.items()):
                time_steps = sorted(data[metric].keys())
                metric_values = [data[metric][t] for t in time_steps]
                
                # If metric values are auc_pr or recall, subtract them from 1 to get 1 - AUC-PR or 1 - RECALL
                if metric == 'auc_pr' or metric == 'recall':
                    metric_values = [1 - v for v in metric_values]
                    
                
                marker_style = marker_styles[i % len(marker_styles)]

                slabel = 100 - stochasticity_level
                plt.plot(time_steps, metric_values, label=slabel, 
                        color=colors[i % len(colors)], marker=marker_style, linestyle='-', linewidth=2.5, 
                        markersize=4, markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=colors[i % len(colors)])

            plt.xlabel('Time Step', fontsize=22, fontweight='bold')  # Adjusted font size and weight
            if metric == 'auc_pr':
                plt.ylabel('1 - AUC-PR', fontsize=22, fontweight='bold')  # Adjusted font size and weight
            elif metric == 'recall':
                plt.ylabel('1 - RECALL', fontsize=22, fontweight='bold')
            else:
                plt.ylabel(metric.upper(), fontsize=22, fontweight='bold')  # Adjusted font size and weight
            # plt.title(f'{metric.capitalize()} over Time Steps', fontsize=14, fontweight='bold')  # Adjust title fontsize and weight

            plt.xticks(fontsize=22)  # Adjusted font size
            plt.yticks(fontsize=22)  # Adjusted font size

            legend = plt.legend(title='S-Level', title_fontsize='12', fontsize=16, 
                            frameon=True, edgecolor='black', ncol=5, loc='upper center',
                            bbox_to_anchor=(0.5, 1.5))  # Adjusted legend configuration
            
            def legend_title_left(leg):
                c = leg.get_children()[0]
                title = c.get_children()[0]
                hpack = c.get_children()[1]
                c._children = [hpack]
                hpack._children = [title] + hpack.get_children()
            legend_title_left(legend) # Move the legend title to the left
            
            plt.setp(legend.get_title(), fontweight='bold', fontsize=18)  # Adjust title weight and size
            plt.setp(legend.get_texts(), fontweight='bold', fontsize=18)  # Adjust text weight and size

            # Set ylim for different metrics: AUC-PR and Recall (0.75:1.01), ECE (0.0:0.25), MSE (0.0:0.25)
            if metric == 'auc_pr' or metric == 'recall':
                # plt.ylim(0.75, 1.01)
                pass
            elif metric == 'ece':
                plt.ylim(-0.01, 0.25)
            elif metric == 'mse':
                plt.ylim(-0.01, 0.25)
                    
            
            # plt.ylim(0.70, 1.01)  # Set the upper y-axis limit

            plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.2, zorder=1)  # Adjust grid to match previous style

            plt.tight_layout()  # Adjust layout

            # Save the plot with a similar naming scheme but retaining the PNG format
            filename = os.path.join(savepath, f'{metric}_over_time_steps_combined.pdf')
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
            print(f'Plot saved to {filename}')


def main():
    # Get the location of the folder where the raw predictions and forecastGT are stored
    script_folder = os.path.dirname(os.path.realpath(__file__))
    project_folder = os.path.dirname(os.path.dirname(script_folder))
    forecast_GT_folder = os.path.join(project_folder, "convLSTM_inference_runs")
    
    
    dnn_name = "d-convlstm" or "d-convlstm-slevel-90"
    stochasticity_list = [100, 95, 90, 85, 80]
    base_savepath = "Enter the base path to save the plots"
    os.makedirs(base_savepath, exist_ok=True)

    metric_vs_time_combined_simulation.generate_metrics_combined_simulation(stochasticity_list, dnn_name, base_savepath, dataset_type= "diffInit",
                                                                            num_samples=300,
                                                                            forecast_GT_folder=forecast_GT_folder)
    
    
if __name__ == '__main__':
    main()