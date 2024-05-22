"""
Used for creating Table 1 and Table 2 in the paper.
"""

# Generates the table of scores for the next day wildfire dataset for the paper
from baseline_vs_proposed_score_compare import get_scores_from_json
from baseline_vs_proposed_score_compare import MC_Score_Plotter_stochasticity_stratified
import numpy as np
import pickle
import calibration as cal
import json
import matplotlib.pyplot as plt
from stochasticity_estimator_nextdaywildfire import create_mask, jaccard_similarity, dice_similarity
import matplotlib.colors as mcolors
from sklearn.metrics import precision_score, recall_score, mean_squared_error, precision_recall_curve, auc

def plot_images_side_by_side(sample_list):
    """
    Plots prediction, ground truth, and previous day images side by side.
    
    Parameters:
    - sample_list: A list containing prediction, GT, and previous day images.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns
    
    # Create a colormap for the labels and PrevFireMask
    cmap = mcolors.ListedColormap(['grey', 'black', 'white'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Titles for each subplot
    titles = ['Prediction', 'Ground Truth', 'Previous Day']
    
    for i, ax in enumerate(axes):
        # Assuming sample_list contains 2D NumPy arrays for each image
        img = np.array(sample_list[i])
        
        if titles[i] == "Prediction":
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img, cmap=cmap, norm=norm)  
        ax.set_title(titles[i])
        ax.axis('off')  # Hide axes ticks

    plt.tight_layout()
    plt.savefig("sample_images.png")
    plt.close()

def calculate_similarity_metrics(img_t, img_t1):
    metrics = {}
    
    assert img_t.shape == img_t1.shape, f'Input and label shapes do not match ({img_t.shape} vs {img_t1.shape})'
    assert set(np.unique(img_t)).issubset([0, 1, -1]), f'Input image contains values other than 0, 1, and -1'
    assert set(np.unique(img_t1)).issubset([0, 1, -1]), f'Input image contains values other than 0, 1, and -1'
    
    mask = create_mask(img_t, img_t1)
    img_t_masked = img_t[mask]
    img_t1_masked = img_t1[mask]
    assert set(np.unique(img_t_masked)).issubset([0, 1]), f"Input image contains values other than 0, 1"
    assert set(np.unique(img_t1_masked)).issubset([0, 1]), f'Input image contains values other than 0, 1'
        
    # Calculate and store each metric
    metrics['Jaccard_Similarity'] = jaccard_similarity(img_t_masked, img_t1_masked)
    metrics['Dice_Similarity'] = dice_similarity(img_t_masked, img_t1_masked)
    
    return metrics

def convert_bskBaselineOutput_to_stochasticityDict_format(fpath, dnn_name):
    stochasticity_score_dict = {}
    
    # Read the json file at fpath
    with open(fpath, "r") as f:
        data = json.load(f)
    
    for sample_idx, sample_list in data.items():
        # Convert the list to numpy arrays
        sample_list = [np.array(sample) for sample in sample_list]
        prediction = sample_list[0]
        gt = sample_list[1]
        prev_day = sample_list[2]
        
        
        # Get the stochasticity estimates
        overlap_metrics = calculate_similarity_metrics(img_t=prev_day, img_t1=gt)
        stochasticity_score_dict[sample_idx] = {"score": None, "stochasticity": overlap_metrics['Dice_Similarity'],
                                                    "target_sample": gt, 
                                                    "output_sample": prediction}

        # # For viewing the sample
        # print(f"Sample {sample_idx}: prediction: {prediction.shape}, GT: {gt.shape}, prevDay: {prev_day.shape}, DC: {overlap_metrics['Dice_Similarity']}")
        # print(np.unique(gt), np.unique(prev_day))
        # plot_images_side_by_side(sample_list)
        # input("Press Enter to continue...")
        
        
    return stochasticity_score_dict

def create_score_stochasticity_dict(metric, scale, score_file_path, stochasticity_file_path, pred_gt_file_path):
        """
        Returns a dictionary containing the scores and stochasticity estimates for each sample.
        Format: {sample_idx: {"score": score_val, "stochasticity": stochasticity_dict}} 
        """
        # Load the scores
        # score_file_path = f"/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/raw_evaluation_scores/eval_segmentation_model_300epochs_MC_run{MC_run}__bottleneck{bottleneck_flag}_kernelSize3_bottleneckChannel{bottleneck_channel}/raw_scores.json"    
        scores_list = get_scores_from_json(score_file_path, scale, metric)
        
        # Load the stochasticity estimates
        # stochasticity_file_path = f"/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/stochasticity_estimates/eval_segmentation_model_300epochs_MC_run{MC_run}__bottleneck{bottleneck_flag}_kernelSize3_bottleneckChannel{bottleneck_channel}/stochasticity_estimates.json"
        stochasticity_dict = MC_Score_Plotter_stochasticity_stratified.load_stochasticity_estimates(stochasticity_file_path)
        
        # Load the predictions and ground truth from the pickle file
        with open(pred_gt_file_path, "rb") as f:
            pred_gt_dict = pickle.load(f)
        
        ########################### Order is swapped in the pickle file (due to bug) ###########################
        # target_sample_dict = pred_gt_dict["output_sample_dict"] 
        # output_sample_dict = pred_gt_dict["target_sample_dict"]
        
        target_sample_dict = pred_gt_dict["target_sample_dict"] 
        output_sample_dict = pred_gt_dict["output_sample_dict"]
        
        
        # Merge the scores, stochasticity estimates and the target and output samples into a single dictionary
        stochasticity_score_dict = {}
        for sample_idx, score_val in enumerate(scores_list):
            stochasticity_score_dict[sample_idx] = {"score": score_val, "stochasticity": stochasticity_dict[str(sample_idx)]['Dice_Similarity'],
                                                    "target_sample": target_sample_dict[sample_idx], 
                                                    "output_sample": output_sample_dict[sample_idx]}
            
            # # For debugging purposes
            # plot_images_side_by_side([output_sample_dict[sample_idx], target_sample_dict[sample_idx], target_sample_dict[sample_idx]])
            # print(np.unique(output_sample_dict[sample_idx]))
            # print(np.unique(target_sample_dict[sample_idx]))
            # input("Press Enter to continue...")
        
        return stochasticity_score_dict
    

def combine_samples_based_on_stochasticity_bins(stochasticity_score_dict, num_bins=10):
    # Initialize a dictionary to hold combined target and output arrays for each bin
    combined_data_by_bin = {bin_idx: {'targets': [], 'outputs': [], 'num_samples':0} for bin_idx in range(num_bins)}
    
    for idx,sample_dict in enumerate(stochasticity_score_dict.values()):
        # print(f"Sample: {idx}")
        # Determine the bin index for the current sample's stochasticity
        bin_idx = int(sample_dict['stochasticity'] * num_bins)
        bin_idx = min(bin_idx, num_bins - 1)  # Ensure the maximum index is within bounds
        
        output_t = sample_dict['output_sample']
        target_t = sample_dict['target_sample']
        output_b_flat = output_t.flatten()  # Shape: (1, H, W) -> (H * W)
        target_b_flat = target_t.flatten()  # Shape: (1, H, W) -> (H * W)
        
        # Remove the uncertain pixels from the output and target tensors
        mask = (target_b_flat != -1)
        output_b_flat = output_b_flat[mask]
        target_b_flat = target_b_flat[mask]
        assert output_b_flat.shape == target_b_flat.shape, f"Shape of outputs and targets should be the same across scales {output_b_flat.shape} != {target_b_flat.shape}"
        
        # Flatten the target and output samples and append to the respective lists
        combined_data_by_bin[bin_idx]['targets'].append(target_b_flat)
        combined_data_by_bin[bin_idx]['outputs'].append(output_b_flat)
        
        # Increment the number of samples in the bin
        combined_data_by_bin[bin_idx]['num_samples'] += 1
    
    # Combine (concatenate) and store flattened arrays for targets and outputs in each bin
    for bin_idx, data in combined_data_by_bin.items():
        combined_data_by_bin[bin_idx]['targets'] = np.concatenate(data['targets'], axis=0).astype(int) if data['targets'] else np.array([])
        combined_data_by_bin[bin_idx]['outputs'] = np.concatenate(data['outputs'], axis=0) if data['outputs'] else np.array([])
    
    return combined_data_by_bin


def generate_stratified_ece_table(metric, scale, score_file_path, stochasticity_file_path, pred_gt_file_path, stochasticity_score_dict=None, num_bins=10):
    if stochasticity_score_dict is None:
        # Get the dictionary containing the scores and stochasticity estimates for each sample
        stochasticity_score_dict = create_score_stochasticity_dict(metric, scale, score_file_path, stochasticity_file_path, pred_gt_file_path)
    
    # Combine samples based on stochasticity bins
    combined_data_by_bin = combine_samples_based_on_stochasticity_bins(stochasticity_score_dict, num_bins=num_bins)
    
    # Calculate metrics for each bin
    ece_by_bin = {}
    mse_by_bin = {}
    recall_by_bin = {}
    precision_by_bin = {}
    aucPR_by_bin = {}
    
    for bin_idx, data in combined_data_by_bin.items():
        # print("**********************")
        # print(f"Targets: {np.unique(data['targets'])}")
        # print(f"Outputs: {np.unique(data['outputs'])}")
        target_b_flat = data['targets']
        output_b_flat = data['outputs']
        
        ece_by_bin[bin_idx] = cal.get_ece(output_b_flat, target_b_flat)
        precision_by_bin[bin_idx] = precision_score(target_b_flat, (output_b_flat > 0.5).astype(int), zero_division=0)
        recall_by_bin[bin_idx] = recall_score(target_b_flat, (output_b_flat > 0.5).astype(int), zero_division=0)
        mse_by_bin[bin_idx] = mean_squared_error(target_b_flat, output_b_flat)
        
        # ROC PR curve
        precision_array, recall_array, _ = precision_recall_curve(target_b_flat, output_b_flat)
        aucPR_by_bin[bin_idx] = auc(recall_array, precision_array)
            
    # Pretty print all the metrics (order should be precision, recall, aucpr, mse, ece)
    for bin_idx, ece_val in ece_by_bin.items():
        print(f"Bin {bin_idx} | #samples {combined_data_by_bin[bin_idx]['num_samples']} -> Precision: {precision_by_bin[bin_idx]:.3f}, Recall: {recall_by_bin[bin_idx]:.3f}, AUC PR: {aucPR_by_bin[bin_idx]:.3f}, MSE: {mse_by_bin[bin_idx]:.3f}, ECE: {ece_val:.3f}")        
        
        
def generate_baselines_table(dnn_name="segformer", num_bins=10):
    dnn_name_path_dict = {"segformer":"/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/baselines_next_day/segformer_result.json",
                          "arnca":"/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/baselines_next_day/arnca_result.json",
                          "unet":"/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/baselines_next_day/unet_result.json",
                          "segnet":"/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/baselines_next_day/segnetv2_result.json"}
    stochasticity_score_dict = convert_bskBaselineOutput_to_stochasticityDict_format(fpath=dnn_name_path_dict[dnn_name], dnn_name=dnn_name)
    
    generate_stratified_ece_table(metric=None, 
                                  scale=None, 
                                  score_file_path=None, 
                                  stochasticity_file_path=None, 
                                  pred_gt_file_path=None, 
                                  stochasticity_score_dict=stochasticity_score_dict,
                                  num_bins=num_bins)
    
def main():
    metric = "recall_scores"
    scale = "1"
    
    # For bottleneck huot
    score_file_path = "/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/raw_evaluation_scores/eval_ECE_segmentation_model_300epochs_MC_run10__bottleneckTrue_kernelSize3_bottleneckChannel16/raw_scores.json"
    stochasticity_file_path = "/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/stochasticity_estimates/eval_ECE_segmentation_model_300epochs_MC_run10__bottleneckTrue_kernelSize3_bottleneckChannel16/stochasticity_estimates.json"
    pred_gt_file_path = "/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/raw_evaluation_scores/eval_ECE_segmentation_model_300epochs_MC_run10__bottleneckTrue_kernelSize3_bottleneckChannel16/sample_output_target.pkl"
    
    # # For no bottleneck huot
    # score_file_path = "/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/raw_evaluation_scores/eval_ECE_segmentation_model_300epochs_MC_run28__bottleneckFalse_kernelSize3_bottleneckChannel16/raw_scores.json"
    # stochasticity_file_path = "/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/stochasticity_estimates/eval_ECE_segmentation_model_300epochs_MC_run28__bottleneckFalse_kernelSize3_bottleneckChannel16/stochasticity_estimates.json"
    # pred_gt_file_path = "/data/hkumar64/projects/starcraft/chaosnet/next_day_wildfire_spread/dump/raw_evaluation_scores/eval_ECE_segmentation_model_300epochs_MC_run28__bottleneckFalse_kernelSize3_bottleneckChannel16/sample_output_target.pkl"
        
    # Generate the stratified ECE table
    generate_stratified_ece_table(metric, scale, score_file_path, stochasticity_file_path, pred_gt_file_path, stochasticity_score_dict=None, num_bins=1)
    exit()
    
    # Generate the baseline scores table
    dnn_name = "arnca"         # "segformer","arnca","unet","segnet"
    generate_baselines_table(dnn_name=dnn_name, num_bins=1)
    
    # for dnn_name in ["segformer","arnca","unet","segnet"]:
    #     print(f"Generating table for {dnn_name}")
    #     generate_baselines_table(dnn_name=dnn_name)
    
if __name__ == "__main__":
    main()