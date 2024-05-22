"""
Used for creating Figure 16 in the paper
"""
from ece_score_table_generator import combine_samples_based_on_stochasticity_bins, convert_bskBaselineOutput_to_stochasticityDict_format, create_score_stochasticity_dict
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import os
import matplotlib.ticker as mticker
import numpy as np

def load_combined_data(dnn_name):
    """
    Combined data stores all the combined target and output values for each bin
    """
    # Load the stochasticity_score_dict for each dnn_name
    dnn_name_path_dict = {"segformer":"/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/baselines_next_day/segformer_result.json",
                          "arnca":"/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/baselines_next_day/arnca_result.json",
                          "unet":"/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/baselines_next_day/unet_result.json",
                          "segnet":"/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/baselines_next_day/segnetv2_result.json"}
    if dnn_name in ["segformer","arnca","unet","segnet"]:
        stochasticity_score_dict = convert_bskBaselineOutput_to_stochasticityDict_format(fpath=dnn_name_path_dict[dnn_name], dnn_name=dnn_name)
    elif dnn_name in ["Conv-AE","Conv-CA"]:
        # Filler values to not break the code
        metric = "recall_scores"
        scale = "1"
        if dnn_name == "Conv-AE":
            score_file_path = "/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/raw_evaluation_scores/eval_ECE_segmentation_model_300epochs_MC_run10__bottleneckTrue_kernelSize3_bottleneckChannel16/raw_scores.json"
            stochasticity_file_path = "/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/stochasticity_estimates/eval_ECE_segmentation_model_300epochs_MC_run10__bottleneckTrue_kernelSize3_bottleneckChannel16/stochasticity_estimates.json"
            pred_gt_file_path = "/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/raw_evaluation_scores/eval_ECE_segmentation_model_300epochs_MC_run10__bottleneckTrue_kernelSize3_bottleneckChannel16/sample_output_target.pkl"
        elif dnn_name == "Conv-CA":
            score_file_path = "/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/raw_evaluation_scores/eval_ECE_segmentation_model_300epochs_MC_run28__bottleneckFalse_kernelSize3_bottleneckChannel16/raw_scores.json"
            stochasticity_file_path = "/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/stochasticity_estimates/eval_ECE_segmentation_model_300epochs_MC_run28__bottleneckFalse_kernelSize3_bottleneckChannel16/stochasticity_estimates.json"
            pred_gt_file_path = "/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/raw_evaluation_scores/eval_ECE_segmentation_model_300epochs_MC_run28__bottleneckFalse_kernelSize3_bottleneckChannel16/sample_output_target.pkl"
                    
        stochasticity_score_dict = create_score_stochasticity_dict(metric, 
                                                                   scale, 
                                                                   score_file_path, 
                                                                   stochasticity_file_path, 
                                                                   pred_gt_file_path)

    combined_data_by_bin = combine_samples_based_on_stochasticity_bins(stochasticity_score_dict, num_bins=1)
    return combined_data_by_bin

def main():
    # dnn_names = ["Conv-AE", "Conv-CA", "segformer", "arnca", "unet"]
    dnn_names = ["Conv-AE", "Conv-CA", "segformer", "unet", "arnca"]
    plt.figure(figsize=(8, 6))
    savepath = '/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/calibration_curve_baselines_compare'
    

    for idx, dnn_name in enumerate(dnn_names):
        print(f"Processing {dnn_name}")
        combined_data_by_bin = load_combined_data(dnn_name)    
        target_b_flat = combined_data_by_bin[0]['targets']
        output_b_flat = combined_data_by_bin[0]['outputs']
        
        
        # Plot histogram of all predicted probabilities
        plt.figure(figsize=(8, 6))
        plt.hist(output_b_flat, bins=10, alpha=0.7, color='blue', edgecolor='black', linewidth=1.2, hatch='//', label='DNN Forecast', zorder=3)
        plt.xlabel("DNN Forecast", fontsize=16, fontweight='bold')
        plt.ylabel("Frequency", fontsize=16, fontweight='bold')
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.1, zorder=0)
        plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # plt.ylim(0, 1e5)
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, f'{dnn_name}_predicted_probabilities_histogram.pdf'), bbox_inches='tight', dpi=300)
        plt.close()

        # Plot histogram of predicted probabilities with values >= 0.1
        plt.figure(figsize=(8, 5))
        filtered_probabilities = [p for p in output_b_flat if p >= 0.1]
        plt.hist(filtered_probabilities, bins=10, alpha=0.7, color='blue', label='DNN Forecast', zorder=3, hatch='//', edgecolor='black', linewidth=1.2)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.ylim(0, 1e5)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.1, zorder=0)
        plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, f'{dnn_name}_predicted_probabilities_histogram_exclude_0.1.pdf'), bbox_inches='tight', dpi=300)
        plt.close()

        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(target_b_flat, output_b_flat, n_bins=10)

        # Plotting each model's calibration curve
        plt.plot(mean_predicted_value, fraction_of_positives, label=f'{dnn_name}', marker='o', linestyle='-', zorder=2)
        print(f"Calibration curve plotted for {dnn_name}")

    # Reference line for perfect calibration
    plt.plot([0, 1], [0, 1], "k:", zorder=1)
    plt.xlabel("Mean predicted value", fontsize=16, fontweight='bold')
    plt.ylabel("Fraction of positives", fontsize=16, fontweight='bold')
    plt.legend(fontsize=16, loc='best').get_title().set_fontweight('bold')
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.1, zorder=1)
    plt.ylim(-0.01, 1.02)
    plt.tight_layout()

    # Save the combined calibration curve plot
    os.makedirs(savepath, exist_ok=True)
    plt.savefig(os.path.join(savepath, 'combined_calibration_curve.pdf'), bbox_inches='tight', dpi=300)
    plt.close()
    

if __name__ == '__main__':
    main()