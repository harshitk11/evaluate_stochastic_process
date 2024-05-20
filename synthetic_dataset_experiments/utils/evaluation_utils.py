from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error, log_loss, roc_curve, auc, precision_recall_curve, precision_score, recall_score, confusion_matrix, accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import os
import json
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.calibration import calibration_curve
import calibration as cal


class ScaleWiseEvaluation(nn.Module):
    def __init__(self, scales, config):
        super(ScaleWiseEvaluation, self).__init__()
        self.scales = scales
        self.config = config
        
        # Initialize variables to store cumulative scores
        self.f1_scores = {scale: [] for scale in scales}
        self.roc_auc_scores = {scale: [] for scale in scales}
        self.mse_scores = {scale: [] for scale in scales}  
        self.bce_scores = {scale: [] for scale in scales}  
        self.frame_complexity = {scale: [] for scale in scales}
        self.time_stratified_performance = {scale: {} for scale in scales}
        
    def forward(self, outputs, targets, padding_masks, all_hash_chunk_indices, threshold=0.5, loss_calculation_scales=None, gt_probability_map=None):
        """
        Args:
            outputs: List of tensors containing the output of the model for each scale (B, T, C, H, W)| C = 1
            targets: List of tensors containing the target for each scale (B, T, C, H, W) | C = 1
            padding_masks: Tensor containing the padding mask for each sample (B, T)
            threshold: Threshold for converting the output to binary
            all_hash_chunk_indices: (hash_string, idx, total_chunks) for each sample in the batch
            gt_probability_map: Tensor containing the ground truth probability map if using the same initialization data (B, T', H, W)
        """
        loss_calculation_scales = loss_calculation_scales if loss_calculation_scales else self.scales
        optimal_threshold = {}
        for scale_idx, scale in enumerate(self.scales):
            if scale in loss_calculation_scales:
                optimal_threshold[scale] = {"roc": [], "pr": []}
                
                output = outputs[scale_idx]
                target = targets[scale_idx]
                assert output.shape == target.shape, f"Shape of outputs and targets should be the same across scales {output.shape} != {target.shape}"
                
                padding_mask = padding_masks[:, 1:]
                padding_mask = padding_mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)
                assert output.shape[:2] == padding_mask.shape[:2], f"Shape of outputs and padding masks should be the same across scales {output.shape[:2]} != {padding_mask.shape[:2]}"
                
                # Filter out the batches based on all_hash_chunk_indices (Taking only the first chunk of each hash string)
                batch_indices = [True if x[1] == 0 else False for x in all_hash_chunk_indices]
                output = output[batch_indices]
                target = target[batch_indices]
                padding_mask = padding_mask[batch_indices]        
                    
                # Loop over time steps in T axis
                for t in range(output.shape[1]):
                    # Get non-padded indices
                    non_padded_indices = (padding_mask[:, t] == 1).squeeze()
                    if non_padded_indices.numel() == 1:
                        non_padded_indices = non_padded_indices.unsqueeze(0)
                    
                    # Get non-padded outputs and targets for the current time step
                    output_t = output[non_padded_indices, t]
                    target_t = target[non_padded_indices, t]
                    assert output_t.shape == target_t.shape, f"Shape of outputs and targets should be the same across scales {output_t.shape} != {target_t.shape}"
                    
                    # Calculate and store metrics if there are non-padded frames at the current time step
                    if output_t.numel() > 0 and target_t.numel() > 0:
                        optimal_threshod_roc, optimal_threshod_pr = self.calculate_and_store_metrics(scale, t, output_t, target_t, threshold)
                        optimal_threshold[scale]["roc"].append(optimal_threshod_roc)
                        optimal_threshold[scale]["pr"].append(optimal_threshod_pr)
                        
        return optimal_threshold
    
    def calculate_and_store_metrics(self, scale, t, output_t, target_t, threshold):
        # Initialize lists to store F1 and ROC AUC scores for each batch
        f1_scores_per_batch = []
        roc_auc_scores_per_batch = []
        mse_scores_per_batch = []  
        bce_scores_per_batch = []  
        frame_complexity_per_batch = []
        optimal_threshold_per_batch = []
        optimal_F1_per_batch = []
        
        optimal_threshold_per_batch_pr = []
        optimal_F1_per_batch_pr = []
        optimal_recall_per_batch_pr = []
        optimal_precision_per_batch_pr = []
        
        precision_per_batch = []
        recall_TPR_per_batch = []
        roc_pr_scores_per_batch = []
        fpr_per_batch = []
        accuracy_scores_per_batch = []
        mae_scores_per_batch = []
        ssim_scores_per_batch = []
        
        recall_zero_threshold_per_batch = []
        precision_zero_threshold_per_batch = []
        
        expected_calibration_error_per_batch = []
        
        # Loop over each batch
        for b in range(output_t.shape[0]):
            # Flatten the tensors to calculate metrics for the current batch
            output_b_flat = output_t[b].flatten().cpu().numpy()  # Shape: (1, H, W) -> (H * W)
            target_b_flat = target_t[b].flatten().cpu().numpy()  # Shape: (1, H, W) -> (H * W)
            assert output_b_flat.shape == target_b_flat.shape, f"Shape of outputs and targets should be the same across scales {output_b_flat.shape} != {target_b_flat.shape}"
            
            # Calculate F1 and ROC AUC scores for the current batch
            if np.unique(target_b_flat).size > 1:
                # ignore zero division error
                f1_scores_per_batch.append(f1_score(target_b_flat, (output_b_flat > threshold).astype(int), zero_division=0))
                roc_auc_scores_per_batch.append(roc_auc_score(target_b_flat, output_b_flat))
                mse_scores_per_batch.append(mean_squared_error(target_b_flat, output_b_flat))  
                bce_scores_per_batch.append(log_loss(target_b_flat, output_b_flat))  
                
                # Compute frame complexity
                transformed_probs = np.abs(output_b_flat - 0.5)
                std_dev_complexity = np.std(transformed_probs)
                frame_complexity_per_batch.append(std_dev_complexity)
                
                # Compute optimal threshold and optimal F1 score using roc curve
                optimal_threshold = ScaleWiseEvaluation.find_optimal_threshold_roc_curve(y_true=target_b_flat, y_scores=output_b_flat, save_path=None)
                optimal_F1 = f1_score(target_b_flat, (output_b_flat > optimal_threshold).astype(int))
                optimal_threshold_per_batch.append(optimal_threshold)
                optimal_F1_per_batch.append(optimal_F1)
                
                # Compute optimal threshold and optimal F1 score using Precision-Recall curve
                optimal_threshold_pr = ScaleWiseEvaluation.find_optimal_threshold_pr(y_true=target_b_flat, y_scores=output_b_flat, save_path=None)
                
                optimal_F1_pr = f1_score(target_b_flat, (output_b_flat > optimal_threshold_pr).astype(int), zero_division=0)
                optimal_threshold_per_batch_pr.append(optimal_threshold_pr)
                optimal_F1_per_batch_pr.append(optimal_F1_pr)
                
                optimal_recall_pr = recall_score(target_b_flat, (output_b_flat > optimal_threshold_pr).astype(int))
                optimal_recall_per_batch_pr.append(optimal_recall_pr)
                
                optimal_precision_pr = precision_score(target_b_flat, (output_b_flat > optimal_threshold_pr).astype(int), zero_division=0)
                optimal_precision_per_batch_pr.append(optimal_precision_pr)
                
                # print(f"{t} - original F1: {f1_scores_per_batch[-1]}, optimal F1: {optimal_threshold,optimal_F1_per_batch[-1]}, optimal F1 pr: {optimal_threshold_pr, optimal_F1_per_batch_pr[-1]}")
                
                # Compute precision, recall_TPR, FPR
                precision_per_batch.append(precision_score(target_b_flat, (output_b_flat > threshold).astype(int)))
                recall_TPR_per_batch.append(recall_score(target_b_flat, (output_b_flat > threshold).astype(int)))
                tn, fp, fn, tp = confusion_matrix(target_b_flat, (output_b_flat > threshold).astype(int)).ravel()
                fpr = fp / (fp + tn) if (fp + tn) != 0 else 0  # Handling the case where denominator is zero
                fpr_per_batch.append(fpr)
                
                # ROC PR curve
                precision_array, recall_array, _ = precision_recall_curve(target_b_flat, output_b_flat)
                roc_pr_scores_per_batch.append(auc(recall_array, precision_array))
                
                # Accuracy and MAE        
                accuracy_scores_per_batch.append(accuracy_score(target_b_flat, (output_b_flat > threshold).astype(int)))
                mae_scores_per_batch.append(mean_absolute_error(target_b_flat, output_b_flat))
                
                # Add SSIM
                ssim_scores_per_batch.append(compare_ssim(target_b_flat, output_b_flat, win_size=11, data_range=1))
                
                # Recall and precision at zero threshold
                recall_zero_threshold_per_batch.append(recall_score(target_b_flat, (output_b_flat > 0).astype(int)))
                precision_zero_threshold_per_batch.append(precision_score(target_b_flat, (output_b_flat > 0).astype(int)))
                
                # Expected Calibration Error
                y_true = target_b_flat.astype(int)
                y_prob = output_b_flat
                ece_custom = cal.get_ece(y_prob, y_true) 
                expected_calibration_error_per_batch.append(ece_custom)
                
                
        # Calculate average scores for the current time step
        avg_f1_score = np.mean(f1_scores_per_batch)
        avg_roc_auc_score = np.mean(roc_auc_scores_per_batch)
        avg_mse_score = np.mean(mse_scores_per_batch)  
        avg_bce_score = np.mean(bce_scores_per_batch)
        
        avg_complexity = np.mean(frame_complexity_per_batch)  
        
        avg_optimal_threshold = np.mean(optimal_threshold_per_batch)
        avg_optimal_F1 = np.mean(optimal_F1_per_batch)
        
        avg_optimal_threshold_pr = np.mean(optimal_threshold_per_batch_pr)
        avg_optimal_F1_pr = np.mean(optimal_F1_per_batch_pr)
        avg_optimal_recall_pr = np.mean(optimal_recall_per_batch_pr)
        avg_optimal_precision_pr = np.mean(optimal_precision_per_batch_pr)
        
        avg_precision = np.mean(precision_per_batch)
        avg_recall_TPR = np.mean(recall_TPR_per_batch)
        avg_fpr = np.mean(fpr_per_batch)
        avg_roc_pr_score = np.mean(roc_pr_scores_per_batch)
        
        avg_accuracy_score = np.mean(accuracy_scores_per_batch)
        avg_mae_score = np.mean(mae_scores_per_batch)
        
        avg_ssim_score = np.mean(ssim_scores_per_batch)
        
        avg_recall_zero_threshold = np.mean(recall_zero_threshold_per_batch)
        avg_precision_zero_threshold = np.mean(precision_zero_threshold_per_batch)
        
        avg_expected_calibration_error = np.mean(expected_calibration_error_per_batch)
        
        # Store the average scores
        self.f1_scores[scale].append(avg_f1_score)
        self.roc_auc_scores[scale].append(avg_roc_auc_score)
        self.mse_scores[scale].append(avg_mse_score)  
        self.bce_scores[scale].append(avg_bce_score)
        self.frame_complexity[scale].append(avg_complexity)  

        # Update time-stratified performance metrics
        if t not in self.time_stratified_performance[scale]:
            self.time_stratified_performance[scale][t] = []
        self.time_stratified_performance[scale][t].append((avg_f1_score, avg_roc_auc_score, avg_mse_score, avg_bce_score, 
                                                           avg_complexity, 
                                                           avg_optimal_threshold, avg_optimal_F1, 
                                                           avg_optimal_threshold_pr, avg_optimal_F1_pr,
                                                           avg_precision, avg_recall_TPR, avg_fpr, avg_roc_pr_score,
                                                           avg_accuracy_score, avg_mae_score,
                                                           avg_ssim_score,
                                                           avg_optimal_recall_pr, avg_optimal_precision_pr,
                                                            avg_recall_zero_threshold, avg_precision_zero_threshold,
                                                            avg_expected_calibration_error)) 
        
        return avg_optimal_threshold, avg_optimal_threshold_pr
    
    @staticmethod
    def find_optimal_threshold_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, save_path=None):
        """
        Find the optimal threshold from ROC curve and optionally save the plot.

        Parameters:
        y_true (np.ndarray): Ground truth binary labels.
        y_scores (np.ndarray): Target scores or predicted probabilities for positive class.
        save_path (Optional[str]): Path to save the ROC curve plot. If None, the plot is not saved.

        Returns:
        float: Optimal threshold value.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        distances = np.sqrt((0 - fpr)**2 + (1 - tpr)**2)
        index = np.argmin(distances)
        optimal_threshold = thresholds[index]

        if save_path is not None:
            plt.figure(figsize=(8, 8))
            plt.plot(fpr, tpr, label=f'ROC curve (area = {auc(fpr, tpr):.2f})')
            plt.plot([0, 1], [0, 1], 'k--')
            plt.scatter(fpr[index], tpr[index], marker='o', color='red', label=f'Elbow threshold = {optimal_threshold:.2f}')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic')
            plt.legend(loc="lower right")
            plt.savefig(save_path)
            plt.close()

        return optimal_threshold
    
    @staticmethod
    def find_optimal_threshold_pr(y_true: np.ndarray, y_scores: np.ndarray, save_path= None):
        """
        Find the optimal threshold from Precision-Recall curve and optionally save the plot.

        Parameters:
        y_true (np.ndarray): Ground truth binary labels.
        y_scores (np.ndarray): Target scores or predicted probabilities for positive class.
        save_path (Optional[str]): Path to save the Precision-Recall curve plot. If None, the plot is not saved.

        Returns:
        float: Optimal threshold value.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        f1_scores = np.nan_to_num(f1_scores)
        index = np.argmax(f1_scores)
        optimal_threshold = thresholds[index]

        if save_path is not None:
            plt.figure(figsize=(8, 8))
            plt.plot(recall, precision, label=f'Precision-Recall curve')
            plt.scatter(recall[index], precision[index], marker='o', color='red', label=f'Best Threshold = {optimal_threshold:.2f}')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title('Precision-Recall Curve')
            plt.legend(loc="upper right")
            plt.savefig(save_path)

        return optimal_threshold
    
    def get_overall_scores(self):
        overall_scores = {
            "f1_scores": {scale: sum(scores) / len(scores) if scores else 0 for scale, scores in self.f1_scores.items()},
            "roc_auc_scores": {scale: sum(scores) / len(scores) if scores else 0 for scale, scores in self.roc_auc_scores.items()},
            "mse_scores": {scale: sum(scores) / len(scores) if scores else 0 for scale, scores in self.mse_scores.items()}, 
            "bce_scores": {scale: sum(scores) / len(scores) if scores else 0 for scale, scores in self.bce_scores.items()},
            "frame_complexity": {scale: sum(scores) / len(scores) if scores else 0 for scale, scores in self.frame_complexity.items()}  
        }
        
        return overall_scores

    def _plot_metrics(self, metric_data, metric_name, save_path=None, save_path_txt=None, start_timestep=0):
        plt.figure(figsize=(10, 6), dpi=100)
        data_to_save = {}

        for scale in metric_data:
            # Filter the data based on start_timestep
            filtered_data = [(t, mean, std_dev) for t, mean, std_dev in zip(*metric_data[scale]) if t >= start_timestep]
            if not filtered_data:
                continue  # Skip this scale if there are no data points after filtering
            
            # Unpack the filtered data
            time_steps, mean_scores, std_dev_scores = zip(*filtered_data)
            
            # Plot and save the data
            if time_steps:  # Check if data exists for the scale after filtering
                plt.plot(time_steps, mean_scores, marker='o', label=f'Scale: {scale} - {metric_name}')
                plt.fill_between(time_steps, np.array(mean_scores) - np.array(std_dev_scores), np.array(mean_scores) + np.array(std_dev_scores), alpha=0.1)
                
                # Create a JSON-serializable representation of the plot data for the current scale
                data_to_save[scale] = {"time_steps": time_steps, "scores": mean_scores, "std_dev": std_dev_scores}

        plt.xlabel('Time Step', fontsize=14)
        plt.ylabel(metric_name, fontsize=14)
        plt.title(f'{metric_name} Across Scales', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close()

        # Save the JSON representation to a txt file if save_path_txt is provided
        if save_path_txt:
            with open(save_path_txt, "w") as txt_file:
                json.dump(data_to_save, txt_file, indent=4)

    def export_raw_metric_data(self, save_path):
        """
        Export the raw metric data to a JSON file.
        
        Parameters:
        - save_path: The path where the JSON file should be saved.
        """
        # Preparing data dictionary
        raw_metric_data = {}

        for metric_idx, metric_name in enumerate(['F1', 'ROC_AUC', 'MSE', 'BCE', 
                                                "Frame_Complexity", 
                                                "Optimal_Threshold", "Optimal_F1", 
                                                "Optimal_Threshold_PR", "Optimal_F1_PR",
                                                "Precision", "Recall", "FPR", "ROC_PR",
                                                "Accuracy", "MAE",
                                                "SSIM",
                                                "Optimal_Recall_PR", "Optimal_Precision_PR",
                                                "Recall_Zero_Threshold", "Precision_Zero_Threshold",
                                                "Expected_Calibration_Error"]):
            raw_metric_data[metric_name] = {}
            for scale in self.scales:
                if self.time_stratified_performance[scale]:
                    raw_metric_data[metric_name][scale] = {}
                    for t in self.time_stratified_performance[scale].keys():
                        raw_metric_data[metric_name][scale][t] = [float(x[metric_idx]) for x in self.time_stratified_performance[scale][t]]

        with open(os.path.join(save_path, 'raw_metric_data.json'), 'w') as json_file:
            json.dump(raw_metric_data, json_file, indent=4)

    def plot_time_stratified_performance(self, save_path=None):
        plt.style.use('seaborn-whitegrid')

        # Prepare data
        time_steps_data = {scale: list(self.time_stratified_performance[scale].keys()) for scale in self.scales if self.time_stratified_performance[scale]}

        # Added lines: Calculate mean and std dev for each metric at each time step
        metric_data = {}
        for metric_idx, metric_name in enumerate(['F1', 'ROC_AUC', 'MSE', 'BCE', 
                                                  "Frame_Complexity", 
                                                  "Optimal_Threshold", "Optimal_F1", 
                                                  "Optimal_Threshold_PR", "Optimal_F1_PR",
                                                  "Precision", "Recall", "FPR", "ROC_PR",
                                                  "Accuracy", "MAE",
                                                  "SSIM",
                                                  "Optimal_Recall_PR", "Optimal_Precision_PR",
                                                  "Recall_Zero_Threshold", "Precision_Zero_Threshold",
                                                  "Expected_Calibration_Error"]):
            metric_data[metric_name] = {scale: 
                                        (time_steps_data[scale], 
                                        [float(np.mean([x[metric_idx] for x in self.time_stratified_performance[scale][t]])) for t in time_steps_data[scale]],  # mean
                                        [float(np.std([x[metric_idx] for x in self.time_stratified_performance[scale][t]])) for t in time_steps_data[scale]])  # std dev
                                        for scale in self.scales if self.time_stratified_performance[scale]}
        
        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)
        
        # Save the raw data to a JSON file
        self.export_raw_metric_data(save_path)
                    
        # Plot and save metrics
        for metric_name in metric_data:
            if save_path:
                self._plot_metrics(metric_data[metric_name], 
                                   metric_name, 
                                   os.path.join(save_path, f'{metric_name}_scores_across_scales.png'), 
                                   os.path.join(save_path, f'{metric_name}_scores_across_scales.json'), 
                                   start_timestep=10)

    def reset_scores(self):
        self.f1_scores = {scale: [] for scale in self.scales}
        self.roc_auc_scores = {scale: [] for scale in self.scales}
        self.time_stratified_performance = {scale: {} for scale in self.scales}


def consolidate_metrics_across_scales(metric_name, experiment_prefix, save_path=None):
    print(f"Consolidating {metric_name}")
    os.makedirs(save_path, exist_ok=True)
    load_path = "/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/evaluator"
    p_list = [100, 95, 90, 85, 80]
    train_eval_scale_list = [1]
        
    data_dict = {}

    for p in p_list:
        for _scale in train_eval_scale_list:
            if metric_name in ["F1", "ROC_AUC","MSE","BCE","Frame_Complexity","Optimal_Threshold","Optimal_F1","Optimal_Threshold_PR","Optimal_F1_PR","Precision","Recall","FPR","ROC_PR","Accuracy","MAE", "SSIM", "Optimal_Recall_PR", "Optimal_Precision_PR", "Recall_Zero_Threshold", "Precision_Zero_Threshold", "Expected_Calibration_Error"]:
                folder_name = f"{experiment_prefix}_pyramid_v2_stochastic_multiscale_ablation__d72_p{p}_train_{_scale}_eval_{_scale}"
                json_path = os.path.join(load_path, folder_name, f"{metric_name}_scores_across_scales.json")
            elif metric_name in ["jsd", "mse","ssim"]:
                folder_name = f"{experiment_prefix}_pyramid_v2_stochastic_multiscale_ablation__d72_p{p}_train_{_scale}_eval_{_scale}"
                json_path = os.path.join(load_path, folder_name, f"time_stratified_{metric_name}_starting_10.json")
            
            with open(json_path) as f:
                data = json.load(f)

            for scl, scale_info in data.items():
                time_steps = scale_info["time_steps"]
                score = scale_info["scores"] 
                std_dev = scale_info["std_dev"]
                
                scale = _scale

                if scale != _scale:
                    raise ValueError(f"Scale mismatch: {scale} != {_scale}")

                if p not in data_dict:
                    data_dict[p] = {}
                if scale not in data_dict[p]:
                    data_dict[p][scale] = {"time_steps": time_steps, "scores": score, "std_dev": std_dev}

    plt.style.use('classic')  # Use classic style
    colors = plt.get_cmap('tab10').colors  # Use tab10 colormap for a professional look

    # Define marker styles and create an iterator for them
    marker_styles = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', 'X', 'd', '|', '_']
    marker_iter = iter(marker_styles)

    for scale_idx, scale in enumerate(train_eval_scale_list):
        plt.figure(figsize=(7, 5))  # Set the figure size as in the derivative plot

        for p_idx, p in enumerate(p_list):
            time_steps = data_dict[p][scale]["time_steps"]
            scores = data_dict[p][scale]["scores"]
            std_dev = data_dict[p][scale]["std_dev"]
            stochasticity = 100 - p
            marker_style = next(marker_iter)  # Use the next marker style
            print(scores)
            plt.plot(time_steps, scores, label=f'{stochasticity}', marker=marker_style, linewidth=2.5, color=colors[p_idx % len(colors)],
                    markersize=4, markerfacecolor='white', markeredgewidth=1.5, markeredgecolor=colors[p_idx % len(colors)])
            plt.fill_between(time_steps, [score - deviation for score, deviation in zip(scores, std_dev)], 
                            [score + deviation for score, deviation in zip(scores, std_dev)], color=colors[p_idx % len(colors)], alpha=0.1)

        plt.xlabel('Time Steps', fontsize=12, fontweight='bold')  # Font size for xlabel as in derivative plot
        
        if metric_name == "ROC_PR":
            plt.ylabel(f'Area Under the PR Curve', fontsize=12, fontweight='bold')  # Font size for ylabel as in derivative plot
        else:
            plt.ylabel(f'{metric_name}', fontsize=12, fontweight='bold')  # Font size for ylabel as in derivative plot

        plt.xticks(fontsize=14)  # Font size for xticks as in derivative plot
        plt.yticks(fontsize=14)  # Font size for yticks as in derivative plot

        legend = plt.legend(title='S-Level', title_fontsize='12', fontsize=10, 
                            frameon=True, edgecolor='black', ncol=5, loc='upper center',
                            bbox_to_anchor=(0.5, 1.12))

        plt.setp(legend.get_title(), fontweight='bold', fontsize=14)
        plt.setp(legend.get_texts(), fontweight='bold', fontsize=14)
        def legend_title_left(leg):
            c = leg.get_children()[0]
            title = c.get_children()[0]
            hpack = c.get_children()[1]
            c._children = [hpack]
            hpack._children = [title] + hpack.get_children()
        legend_title_left(legend) # Move the legend title to the left
        # Put the legend outside the plot
        
        # set uppler y lim to 1.02 and no lower limit
        plt.ylim(None, 1.01)

        # Set the grid to be behind plot elements with transparency
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.2, zorder=1)

        plt.tight_layout()
        fpath1 = os.path.join(save_path, f"{metric_name}_scale{scale}.pdf")
        plt.savefig(fpath1, bbox_inches='tight', dpi=300)
        plt.close()
        
        
        
        
    
        plt.figure()
        for p in p_list:
            time_steps = data_dict[p][scale]["time_steps"]
            scores = data_dict[p][scale]["scores"]
            std_dev = data_dict[p][scale]["std_dev"]
            stochasticity = 100 - p
            plt.plot(time_steps, std_dev, label=f'Stochasticity: {stochasticity}%', marker='o')
            
        
        plt.title(f'Scores Across Stochasticities at Scale: {scale}')
        plt.xlabel('Time Steps')
        plt.ylabel(f'Std Dev {metric_name} Score')
        plt.legend()
        
        plt.grid(True)
        
        plt.tight_layout()
        fpath1 = os.path.join(save_path, f"std_{metric_name}_scale{scale}.png")
        plt.savefig(fpath1, dpi=300)
        plt.close()        
        


def main():
    # eval_experiment_prefix = "evalSameInit" # "evalSameInit" "evalDiffInit"
    eval_experiment_prefix = "evalDiffInit"
    folder_name = "evaluation" 
    save_path = f"/data/hkumar64/projects/starcraft/chaosnet/chaos-net/dump/consolidate_scores/{folder_name}/{eval_experiment_prefix}"
    os.makedirs(save_path, exist_ok=True)
    # consolidate_metrics_across_scales(metric_name="F1", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="ROC_AUC", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="BCE", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="MSE", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # # consolidate_metrics_across_scales(metric_name="jsd", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="mse", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="ssim", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="Frame_Complexity", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="Optimal_Threshold", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="Optimal_Threshold_PR", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="Optimal_F1", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="Optimal_F1_PR", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    consolidate_metrics_across_scales(metric_name="ROC_PR", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="FPR", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    consolidate_metrics_across_scales(metric_name="Recall", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="Precision", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="Accuracy", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="MAE", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="SSIM", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="Optimal_Recall_PR", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="Optimal_Precision_PR", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="Recall_Zero_Threshold", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="Precision_Zero_Threshold", save_path=save_path, experiment_prefix=eval_experiment_prefix)
    # consolidate_metrics_across_scales(metric_name="Expected_Calibration_Error", save_path=save_path, experiment_prefix=eval_experiment_prefix)

if __name__ == "__main__":
    
    main()