from sklearn.metrics import f1_score, roc_auc_score, mean_squared_error, log_loss, roc_curve, auc, precision_recall_curve, precision_score, recall_score, confusion_matrix, accuracy_score, mean_absolute_error
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import os
import json
from skimage.metrics import structural_similarity as compare_ssim
import calibration as cal
import pickle
from sklearn.calibration import calibration_curve

class ScaleWiseEvaluation(nn.Module):
    def __init__(self, scales, config):
        super(ScaleWiseEvaluation, self).__init__()
        self.scales = scales
        self.config = config
        
        # Initialize variables to store cumulative scores
        self.precision_scores = {scale: [] for scale in scales}
        self.recall_scores = {scale: [] for scale in scales}
        self.f1_scores = {scale: [] for scale in scales}
        self.accuracy_scores = {scale: [] for scale in scales}
        self.roc_auc_scores = {scale: [] for scale in scales}
        self.mse_scores = {scale: [] for scale in scales}  
        self.bce_scores = {scale: [] for scale in scales}  

        self.optimal_precision_scores = {scale: [] for scale in scales}
        self.optimal_recall_scores = {scale: [] for scale in scales}
        self.optimal_f1_scores = {scale: [] for scale in scales}
        
        self.auc_pr_scores = {scale: [] for scale in scales}
        self.fpr_scores = {scale: [] for scale in scales}
        self.mae_scores = {scale: [] for scale in scales}
        
        # Record all the outputs and targets for calculating overall scores
        self.target_all = {scale: [] for scale in scales}
        self.output_all = {scale: [] for scale in scales}
        self.auc_pr_score_all = {scale: [] for scale in scales}
        self.precision_score_all = {scale: [] for scale in scales}
        self.recall_score_all = {scale: [] for scale in scales}
        
        # Expected Calibration Error
        self.ece_custom_scores = {scale: [] for scale in scales}
        self.ece_custom_all = {scale: [] for scale in scales}
        
        # Calibration curve 
        self.calibrationCurve_fraction_of_positives = None
        self.calibrationCurve_mean_predicted_value = None
        self.all_predicted_probabilities = None
        
        # Record the output and target for each sample for visualization
        self.target_sample_dict = {}
        self.output_sample_dict = {}
        
        
    def forward(self, outputs, targets, padding_masks, all_hash_chunk_indices, threshold=0.5, loss_calculation_scales=None, gt_probability_map=None, sample_index=None):
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
                    
                # Loop over time steps in T axis
                for t in range(output.shape[1]):
                    # Get non-padded outputs and targets for the current time step
                    output_t = output[:, t]
                    target_t = target[:, t]
                    assert output_t.shape == target_t.shape, f"Shape of outputs and targets should be the same across scales {output_t.shape} != {target_t.shape}"
                    
                    # Calculate and store metrics if there are non-padded frames at the current time step
                    if output_t.numel() > 0 and target_t.numel() > 0:
                        optimal_threshod_roc, optimal_threshod_pr = self.calculate_and_store_metrics(scale, t, output_t, target_t, threshold, sample_index)
                        optimal_threshold[scale]["roc"].append(optimal_threshod_roc)
                        optimal_threshold[scale]["pr"].append(optimal_threshod_pr)
                        
        return optimal_threshold
    
    def calculate_and_store_metrics(self, scale, t, output_t, target_t, threshold, sample_index=None):
        avg_optimal_threshold, avg_optimal_threshold_pr = None, None
        
        # Assert that batch size is 1
        assert output_t.shape[0] == 1, f"Batch size should be 1, but got {output_t.shape[0]}"
        # Flatten the tensors to calculate metrics for the current batch
        output_b_flat = output_t.flatten().cpu().numpy()  # Shape: (1, H, W) -> (H * W)
        target_b_flat = target_t.flatten().cpu().numpy()  # Shape: (1, H, W) -> (H * W)
        
        # Remove the uncertain pixels from the output and target tensors
        mask = (target_b_flat != -1)
        output_b_flat = output_b_flat[mask]
        target_b_flat = target_b_flat[mask]
        
        assert output_b_flat.shape == target_b_flat.shape, f"Shape of outputs and targets should be the same across scales {output_b_flat.shape} != {target_b_flat.shape}"
        
        # Log the output and target for each sample for visualization
        if sample_index is not None:
            self.target_sample_dict[sample_index] = target_t.squeeze().cpu().numpy()
            self.output_sample_dict[sample_index] = output_t.squeeze().cpu().numpy()
        else:
            raise ValueError("Sample index should be provided for visualization")
            
        if np.unique(target_b_flat).size > 1:
            precision_score_b = precision_score(target_b_flat, (output_b_flat > threshold).astype(int), zero_division=0)
            recall_score_b = recall_score(target_b_flat, (output_b_flat > threshold).astype(int), zero_division=0)
            f1_score_b = f1_score(target_b_flat, (output_b_flat > threshold).astype(int), zero_division=0)
            accuracy_score_b = accuracy_score(target_b_flat, (output_b_flat > threshold).astype(int))
            roc_auc_score_b = roc_auc_score(target_b_flat, output_b_flat)
            mse_score_b = mean_squared_error(target_b_flat, output_b_flat)
            bce_score_b = log_loss(target_b_flat, output_b_flat)  
            
            # Compute optimal threshold and optimal F1 score using Precision-Recall curve
            optimal_threshold_pr = ScaleWiseEvaluation.find_optimal_threshold_pr(y_true=target_b_flat, y_scores=output_b_flat, save_path=None)
            optimal_precision_b = precision_score(target_b_flat, (output_b_flat > optimal_threshold_pr).astype(int), zero_division=0)
            optimal_recall_b = recall_score(target_b_flat, (output_b_flat > optimal_threshold_pr).astype(int), zero_division=0)
            optimal_F1_pr_b = f1_score(target_b_flat, (output_b_flat > optimal_threshold_pr).astype(int), zero_division=0)
            
            # ROC PR curve
            precision_array, recall_array, _ = precision_recall_curve(target_b_flat, output_b_flat)
            auc_pr_score_b = auc(recall_array, precision_array)
            
            # FPR
            tn, fp, fn, tp = confusion_matrix(target_b_flat, (output_b_flat > threshold).astype(int)).ravel()
            fpr_score_b = fp / (fp + tn) if (fp + tn) != 0 else 0  # Handling the case where denominator is zero
            
            # MAE
            mae_score_b = mean_absolute_error(target_b_flat, output_b_flat)
            
            # Expected Calibration Error
            y_true = target_b_flat.astype(int)
            y_prob = output_b_flat
            ece_custom_b = cal.get_ece(y_prob, y_true) 
            
            # Record the outputs and targets
            self.target_all[scale].append(target_b_flat)
            self.output_all[scale].append(output_b_flat)
            
        else:
            # If there is only one class in the target, set all metrics to np.nan
            precision_score_b = np.nan
            recall_score_b = np.nan
            f1_score_b = np.nan
            accuracy_score_b = np.nan
            roc_auc_score_b = np.nan
            mse_score_b = np.nan
            bce_score_b = np.nan
            optimal_precision_b = np.nan
            optimal_recall_b = np.nan
            optimal_F1_pr_b = np.nan
            auc_pr_score_b = np.nan
            fpr_score_b = np.nan
            mae_score_b = np.nan
            ece_custom_b = np.nan
            
        # Store the average scores
        self.precision_scores[scale].append(precision_score_b)
        self.recall_scores[scale].append(recall_score_b)
        self.f1_scores[scale].append(f1_score_b)
        self.accuracy_scores[scale].append(accuracy_score_b)
        self.roc_auc_scores[scale].append(roc_auc_score_b)
        self.mse_scores[scale].append(mse_score_b)  
        self.bce_scores[scale].append(bce_score_b)
        
        self.optimal_precision_scores[scale].append(optimal_precision_b)
        self.optimal_recall_scores[scale].append(optimal_recall_b)
        self.optimal_f1_scores[scale].append(optimal_F1_pr_b)
        
        self.auc_pr_scores[scale].append(auc_pr_score_b)
        self.fpr_scores[scale].append(fpr_score_b)
        self.mae_scores[scale].append(mae_score_b)
        
        self.ece_custom_scores[scale].append(ece_custom_b)
    
        return avg_optimal_threshold, avg_optimal_threshold_pr
    
    
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
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
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
        # Get the overall ROC_PR score using all the outputs and targets
        all_target_flat_scale1 = np.concatenate(self.target_all[1])
        all_output_flat_scale1 = np.concatenate(self.output_all[1])
        print(f"Shape of all_target_flat_scale1: {all_target_flat_scale1.shape}")
        print(f"Shape of all_output_flat_scale1: {all_output_flat_scale1.shape}")
        assert all_target_flat_scale1.shape == all_output_flat_scale1.shape, f"Shape of outputs and targets should be the same across scales {all_target_flat_scale1.shape} != {all_output_flat_scale1.shape}"
        
        # ROC PR curve
        precision_array, recall_array, _ = precision_recall_curve(all_target_flat_scale1, all_output_flat_scale1)
        auc_pr_score_all = auc(recall_array, precision_array)
        optimal_threshold_pr = ScaleWiseEvaluation.find_optimal_threshold_pr(y_true=all_target_flat_scale1, y_scores=all_output_flat_scale1, save_path=None)
        print(f"Optimal threshold: {optimal_threshold_pr}")
        
        # Precision and Recall
        precision_score_all = precision_score(all_target_flat_scale1, (all_output_flat_scale1 > optimal_threshold_pr).astype(int), zero_division=0)
        recall_score_all = recall_score(all_target_flat_scale1, (all_output_flat_scale1 > optimal_threshold_pr).astype(int), zero_division=0)
        
        # ECE
        ece_custom_all = cal.get_ece(all_output_flat_scale1, all_target_flat_scale1.astype(int))
        
        # Calibration curve
        y_true = all_target_flat_scale1.astype(int)
        y_prob = all_output_flat_scale1
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true, y_prob, n_bins=10)
        self.calibrationCurve_fraction_of_positives = fraction_of_positives
        self.calibrationCurve_mean_predicted_value = mean_predicted_value
        self.all_predicted_probabilities = y_prob
        
        overall_scores = {
            "precision_scores": {scale: np.nanmean(scores) if scores else 0 for scale, scores in self.precision_scores.items()},
            "recall_scores": {scale: np.nanmean(scores) if scores else 0 for scale, scores in self.recall_scores.items()},
            "f1_scores": {scale: np.nanmean(scores) if scores else 0 for scale, scores in self.f1_scores.items()},
            "accuracy_scores": {scale: np.nanmean(scores) if scores else 0 for scale, scores in self.accuracy_scores.items()},
            "roc_auc_scores": {scale: np.nanmean(scores) if scores else 0 for scale, scores in self.roc_auc_scores.items()},
            "mse_scores": {scale: np.nanmean(scores) if scores else 0 for scale, scores in self.mse_scores.items()},
            "bce_scores": {scale: np.nanmean(scores) if scores else 0 for scale, scores in self.bce_scores.items()},
            
            "optimal_precision_scores": {scale: np.nanmean(scores) if scores else 0 for scale, scores in self.optimal_precision_scores.items()},
            "optimal_recall_scores": {scale: np.nanmean(scores) if scores else 0 for scale, scores in self.optimal_recall_scores.items()},
            "optimal_f1_scores": {scale: np.nanmean(scores) if scores else 0 for scale, scores in self.optimal_f1_scores.items()},
            
            "auc_pr_scores": {scale: np.nanmean(scores) if scores else 0 for scale, scores in self.auc_pr_scores.items()},
            "fpr_scores": {scale: np.nanmean(scores) if scores else 0 for scale, scores in self.fpr_scores.items()},
            "mae_scores": {scale: np.nanmean(scores) if scores else 0 for scale, scores in self.mae_scores.items()},
            
            "auc_pr_score_all": {1: auc_pr_score_all},
            "precision_score_all": {1: precision_score_all},
            "recall_score_all": {1: recall_score_all},
            
            "ece_custom_scores": {scale: np.nanmean(scores) if scores else 0 for scale, scores in self.ece_custom_scores.items()},
            "ece_custom_all": {1: ece_custom_all},
        }
        
        # Update the overall scores
        self.auc_pr_score_all = {1: auc_pr_score_all}
        self.precision_score_all = {1: precision_score_all}
        self.recall_score_all = {1: recall_score_all}
        self.ece_custom_all = {1: ece_custom_all}
        
        return overall_scores
    
    def save_raw_scores(self, savepath=None):
        scores_dict = {
            "precision_scores": self.precision_scores,
            "recall_scores": self.recall_scores,
            "f1_scores": self.f1_scores,
            "accuracy_scores": self.accuracy_scores,
            "roc_auc_scores": self.roc_auc_scores,
            "mse_scores": self.mse_scores,
            "bce_scores": self.bce_scores,
            "optimal_precision_scores": self.optimal_precision_scores,
            "optimal_recall_scores": self.optimal_recall_scores,
            "optimal_f1_scores": self.optimal_f1_scores,
            "auc_pr_scores": self.auc_pr_scores,
            "fpr_scores": self.fpr_scores,
            "mae_scores": self.mae_scores,
            
            "auc_pr_score_all": self.auc_pr_score_all,
            "precision_score_all": self.precision_score_all,
            "recall_score_all": self.recall_score_all,
            
            "ece_custom_scores": self.ece_custom_scores,
            "ece_custom_all": self.ece_custom_all,
        }
        
        sample_output_target_dict = {   
            "target_sample_dict": self.target_sample_dict,
            "output_sample_dict": self.output_sample_dict,
        }
        
        def json_serializable(item):
            if isinstance(item, np.float32):
                return float(item)
            raise TypeError(f"Type {type(item)} not serializable")
        
        if savepath is not None and not os.path.exists(savepath):
            os.makedirs(savepath)
        filepath = os.path.join(savepath, 'raw_scores.json')
        with open(filepath, 'w', encoding='utf-8') as file:
            json.dump(scores_dict, file, default=json_serializable, ensure_ascii=False, indent=4)
            
        # Save the target and output for each sample as a pickle file
        filepath = os.path.join(savepath, 'sample_output_target.pkl')
        with open(filepath, 'wb') as file:
            pickle.dump(sample_output_target_dict, file)
            

        #################### Calibration Curve ###################
         # Apply the classic Matplotlib style
        plt.style.use('classic')

        # Use the 'tab10' colormap
        # Orange color
        colors = plt.get_cmap('tab10')
        
        plt.figure(figsize=(8, 6))
        plt.plot(self.calibrationCurve_mean_predicted_value, self.calibrationCurve_fraction_of_positives, label=f'ECE = {self.ece_custom_all[1]:.5f}', marker='o', color=colors(1), linewidth=3)
        
        
        plt.plot([0, 1], [0, 1], "k:", zorder=1)  # Reference line for perfect calibration
        plt.xlabel("Mean predicted value", fontsize=20, fontweight='bold')
        plt.ylabel("Fraction of positives", fontsize=20, fontweight='bold')
        
        # plt.title("Calibration Curve", fontsize=16)
        plt.legend(fontsize=22, loc='best').get_title().set_fontweight('bold')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.1, zorder=1)
        plt.ylim(-0.01, 1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, 'calibration_curve.pdf'), bbox_inches='tight', dpi=300)
        plt.close()

        #################### Histogram of Predicted Probabilities ###################
        import matplotlib.ticker as mticker
        plt.figure(figsize=(8, 6))
        # Set a zorder for the histogram to ensure it's drawn above the grid
        # plt.hist(self.all_predicted_probabilities, bins=10, alpha=0.7, color=colors(1), label='DNN Forecast', zorder=3, hatch='/')
        plt.hist(self.all_predicted_probabilities, bins=10, alpha=0.7, color=colors(1), edgecolor='black', linewidth=1.2, hatch='//', label='DNN Forecast', zorder=3)

        plt.xlabel("DNN Forecast", fontsize=20, fontweight='bold')
        plt.ylabel("Frequency", fontsize=20, fontweight='bold')
        # plt.title("Histogram of Predicted Probabilities", fontsize=16)
        # plt.legend(fontsize=14, loc='upper right')
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)

        # Enable grid but set zorder to be lower than the histogram's zorder
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.1, zorder=0)

        # Use exponential notation for y-axis
        plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # To increase the font size of the exponent
        plt.gca().yaxis.get_offset_text().set_size(26)  # Adjust the font size for the exponent

        plt.tight_layout()
        plt.savefig(os.path.join(savepath, 'predicted_probabilities_histogram.pdf'), bbox_inches='tight', dpi=300)
        plt.close()
        
        
        ################### Histogram of predicted probabilities WITH FILTERED VALUES ###################
        plt.figure(figsize=(8, 5))
        filtered_probabilities = [p for p in self.all_predicted_probabilities if p >= 0.1]
        plt.hist(filtered_probabilities, bins=10, alpha=0.7, color=colors(1), label='DNN Forecast', zorder=3, hatch='//', edgecolor='black', linewidth=1.2)
        # plt.xlabel("Predicted probability", fontsize=16)
        # plt.ylabel("Frequency", fontsize=16)
        # plt.title("Histogram of Predicted Probabilities", fontsize=16)
        # plt.legend(fontsize=14, loc='upper right')
        plt.xticks(fontsize=26)
        plt.yticks(fontsize=26)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.1, zorder=0)
        
        # Use exponential notation for y-axis
        plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
        plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
        # To increase the font size of the exponent
        plt.gca().yaxis.get_offset_text().set_size(26)  # Adjust the font size for the exponent


        
        plt.tight_layout()
        plt.savefig(os.path.join(savepath, 'predicted_probabilities_histogram_exclude_0.1.pdf'), bbox_inches='tight', dpi=300)
        plt.close()
        


def main():
    return

if __name__ == "__main__":
    
    main()