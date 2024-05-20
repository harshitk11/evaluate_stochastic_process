"""
Code used to generate the calibration curve on the synthetic dataset (inset in Figure 6 of the paper)
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

# ECE calculation verified against the following formula:
# ece = np.sum(np.abs(fraction_of_positives - mean_predicted_value) * np.histogram(y_prob, bins=10, range=(0,1))[0]) / y_prob.size
# ece = round(ece * 100, 2)  # Converting to percentage and rounding off
"""
## Reference for calibration library: https://pypi.org/project/uncertainty-calibration/

@inproceedings{kumar2019calibration,
  author = {Ananya Kumar and Percy Liang and Tengyu Ma},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  title = {Verified Uncertainty Calibration},
  year = {2019}}
"""

def plot_and_draw_calibration_curves(stochasticity_list, dnn_name, base_savepath, dataset_type, num_samples=1, forecast_GT_folder=None):
    # plt.style.use('classic')  # Apply the classic Matplotlib style
    colors = plt.get_cmap('tab10')  # Get the 'tab10' colormap for consistent color usage
    markers = iter(['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd', '|', '_'])  # Define an iterator for markers

    # Create a subplot with a defined size
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot([0, 1], [0, 1], "k:", zorder=1)  # Plot the reference line for perfect calibration

    for i, stochasticity_level in enumerate(stochasticity_list):
        print(f"Stochasticity Level: {stochasticity_level}")
        # Load the data
        raw_pred_forecastGT_dict = load_raw_pred_forecastGT_dict(dnn_name, stochasticity_level, dataset_type=dataset_type,
                                                                 base_loadpath=forecast_GT_folder)
        
        y_prob = []
        y_true = []
        
        # Collect probabilities and true outcomes
        for j in range(num_samples):
            print(f"Sample: {j}")
            test_sample = raw_pred_forecastGT_dict[j]
            prediction = test_sample['prediction']
            observedGT = test_sample['observedGT']
            
            # Consider only the last time step
            if prediction.shape[0] > 0:
                prediction = prediction[-1]
                observedGT = observedGT[-1]
            
            y_prob.append(prediction.flatten())
            y_true.append(observedGT.flatten())
        
        y_prob = np.concatenate(y_prob)
        y_true = np.concatenate(y_true)
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
        
        y_true = y_true.astype(int)  # Convert y_true to an int array for ECE calculation
        ece = cal.get_ece(y_prob, y_true, num_bins=10) 
        ece = round(ece, 3)  # Round off the ECE value
        # Make sure that 3 decimal places are displayed
        ece = f'{ece:.3f}'
        
        
        updated_label = f'{100-stochasticity_level:2d} [{ece}]'
        # updated_label = f'S-Level {100-stochasticity_level}'
        
        # Plot the calibration curve using the next marker in the sequence
        marker = next(markers)
        ax.plot(mean_predicted_value, fraction_of_positives, marker=marker, linestyle='--', color=colors(i), 
                label=updated_label, markersize=10, zorder=3, linewidth=4)
    
    fontsize_increase = 16
    # Set plot title, labels, and grid style
    ax.set_title('Calibration curve', fontsize=14)
    ax.set_xlabel("Mean predicted value", fontsize=16+fontsize_increase+2)
    ax.set_ylabel("Fraction of positives", fontsize=16+fontsize_increase+2)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.2, zorder=1)
    legend = ax.legend(title='S-Level [ECE]', fontsize=20, title_fontsize=18, loc = 'best')
    ax.get_legend().get_title().set_fontweight('bold')
    plt.setp(legend.get_texts(), fontweight='bold')  # Adjust text weight and size
    
    plt.xticks(fontsize=16+fontsize_increase)
    plt.yticks(fontsize=16+fontsize_increase)
    plt.ylim(-0.01, 1.03)
    
    plt.tight_layout()
    savepath_plots = base_savepath  # Assume base_savepath is defined
    plot_filename = os.path.join(savepath_plots, f'calibration_curve_{dataset_type}.pdf')
    plt.savefig(plot_filename, bbox_inches='tight', dpi=300)
    plt.close()

def main():
    # Get the location of the folder where the raw predictions and forecastGT are stored
    script_folder = os.path.dirname(os.path.realpath(__file__))
    project_folder = os.path.dirname(os.path.dirname(script_folder))
    forecast_GT_folder = os.path.join(project_folder, "convLSTM_inference_runs")
    
    dnn_name = "d-convlstm"
    stochasticity_list = [100, 95, 90, 85, 80]
    base_savepath = "/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/calibration_curve_instant"
    os.makedirs(base_savepath, exist_ok=True)
    
    plot_and_draw_calibration_curves(stochasticity_list, dnn_name, base_savepath, dataset_type='diffInit', num_samples=300,
                                     forecast_GT_folder=forecast_GT_folder)
    

if __name__ == '__main__':
    main()