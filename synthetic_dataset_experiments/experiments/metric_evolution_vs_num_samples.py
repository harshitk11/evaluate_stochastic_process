"""
Code used to generate the evolution of evaluation metrics with the number of samples for different stochasticity levels
(Figure 14)
"""

from reliability_analysis import load_raw_pred_forecastGT_dict, raw_pred_forecastGT_dict
from predictive_difficulty_score_analysis import load_SD_vs_time_step_dict
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os 
from sklearn.calibration import calibration_curve
import calibration as cal
from sklearn.metrics import precision_score, recall_score, precision_recall_curve, auc, mean_squared_error
import pickle

def generate_metrics_evolution(stochasticity_list, dnn_name, base_savepath, dataset_type, num_samples=100, M=10):
    plt.style.use('classic')  # Apply the classic Matplotlib style
    colors = plt.get_cmap('tab10')  # Get the 'tab10' colormap for consistent color usage

    # Initialize dictionaries to store metric values for plotting
    all_metrics = {
        'ece': {},
        'precision': {},
        'recall': {},
        'auc_pr': {},
        'mse': {}
    }

    print(f"Number of Samples: {num_samples}")
    for i, stochasticity_level in enumerate(stochasticity_list):
        print(f"Stochasticity Level: {stochasticity_level}")
        
        # Initialize metric storage for the current stochasticity level with an additional dimension for repetitions
        metrics_vs_samples_reps = {
            'ece': np.zeros((M, num_samples)),
            'precision': np.zeros((M, num_samples)),
            'recall': np.zeros((M, num_samples)),
            'auc_pr': np.zeros((M, num_samples)),
            'mse': np.zeros((M, num_samples))
        }

        for m in range(M):
            print(f"Repetition: {m+1}/{M}")
            raw_pred_forecastGT_dict = load_raw_pred_forecastGT_dict(dnn_name, stochasticity_level, dataset_type=dataset_type)
            
            # Generate a new order for each repetition
            new_order = np.random.permutation(num_samples)
            
            y_prob = []
            y_true = []

            for indx,j in enumerate(new_order):
            # for indx,j in enumerate(range(num_samples)):
                # Print every fifth sample
                if j % 5 == 0:
                    print(f"Sample: {j}")
                    
                test_sample = raw_pred_forecastGT_dict[j]
                prediction = test_sample['prediction']
                observedGT = test_sample['observedGT']
                
                if prediction.shape[0] > 0:
                    prediction = prediction[-1]
                    observedGT = observedGT[-1]
                
                y_prob.append(prediction.flatten())
                y_true.append(observedGT.flatten())

                y_prob_np = np.concatenate(y_prob)
                y_true_np = np.concatenate(y_true).astype(int)
                
                # print(f"{j} - {np.unique(y_true_np, return_counts=True)}")
                
                # Calculate and store the evaluation metrics for the current repetition
                ece_custom = cal.get_ece(y_prob_np, y_true_np)
                precision_val = precision_score(y_true_np, y_prob_np > 0.5)
                recall_val = recall_score(y_true_np, y_prob_np > 0.5)
                mse_val = mean_squared_error(y_true_np, y_prob_np)
                precision_curve, recall_curve, _ = precision_recall_curve(y_true_np, y_prob_np)
                auc_pr_val = auc(recall_curve, precision_curve)

                metrics_vs_samples_reps['ece'][m, indx] = ece_custom
                metrics_vs_samples_reps['precision'][m, indx] = precision_val
                metrics_vs_samples_reps['recall'][m, indx] = recall_val
                metrics_vs_samples_reps['auc_pr'][m, indx] = auc_pr_val
                metrics_vs_samples_reps['mse'][m, indx] = mse_val

        # Calculate mean and variance across repetitions for later plotting
        for metric in all_metrics:
            all_metrics[metric][stochasticity_level] = {
                'mean': np.mean(metrics_vs_samples_reps[metric], axis=0),
                'variance': np.var(metrics_vs_samples_reps[metric], axis=0)
            }

    # Save the all_metrics dictionary as a pickle file
    all_metrics_filename = os.path.join(base_savepath, "pickle_files", f'all_metrics_{dnn_name}_{dataset_type}_{num_samples}_M{M}.pkl')
    os.makedirs(os.path.dirname(all_metrics_filename), exist_ok=True)
    with open(all_metrics_filename, 'wb') as f:
        pickle.dump(all_metrics, f)
    print(f"Metrics saved as {all_metrics_filename}")
    
    return all_metrics
    
def plot_metrics_evolution(stochasticity_list, dnn_name, base_savepath, dataset_type, num_samples=100, M=10):
    # Load from pickle file
    all_metrics_filename = os.path.join(base_savepath, "pickle_files", f'all_metrics_{dnn_name}_{dataset_type}_{num_samples}_M{M}.pkl')
    with open(all_metrics_filename, 'rb') as f:
        all_metrics = pickle.load(f)
    print(f"Metrics loaded from {all_metrics_filename}")
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd', '|', '_']  # Reset iterator for markers
    
    # Plotting
    for metric_name, metrics in all_metrics.items():
        print(f"Plotting {metric_name} evolution")
        fig, ax = plt.subplots(figsize=(8, 5))
        
        for stochasticity_level, metric_values in metrics.items():
            marker = markers[stochasticity_list.index(stochasticity_level)]
            markevery_num = 10 if num_samples > 50 else 1
            x = range(1, num_samples + 1)
            mean_values = metric_values['mean']
            variance_values = metric_values['variance']
            ax.plot(x, mean_values, marker=marker, markevery=markevery_num, linestyle='--', label=f'{100-stochasticity_level}', markersize=6)
            ax.fill_between(x, mean_values - np.sqrt(variance_values), mean_values + np.sqrt(variance_values), alpha=0.2)

        ax.set_xlabel('Number of Samples', fontsize=16)
        ax.set_ylabel(metric_name.upper(), fontsize=16)
        # legned in the middle center top
        ax.legend(title='S-Level', fontsize=10, title_fontsize=16, ncol=5, loc='upper center')
        # Increase legend text size
        for text in ax.get_legend().get_texts():
            text.set_fontsize('16')
        ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.1)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plot_filename = os.path.join(base_savepath, f'{metric_name}_evolution_{dnn_name}_{dataset_type}_{num_samples}_M{M}.pdf')
        plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
        plt.close()



def main():
    dnn_name = "d-convlstm-slevel-90" # "d-convlstm-slevel-90", "d-convlstm"
    stochasticity_list = [100, 95, 90, 85, 80]
    # stochasticity_list = [80]
    base_savepath = "/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/metric_evolution_vs_num_samples/"
    os.makedirs(base_savepath, exist_ok=True)

    # generate_metrics_evolution(stochasticity_list, dnn_name, base_savepath, dataset_type = "diffInit", num_samples=300, M=5)
    plot_metrics_evolution(stochasticity_list, dnn_name, base_savepath, dataset_type = "diffInit", num_samples=300, M=5)

if __name__ == '__main__':
    main()