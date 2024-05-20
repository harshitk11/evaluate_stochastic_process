import torch.nn as nn
import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
import os
import json
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.metrics import mean_squared_error


class JSDivergenceEvaluation(nn.Module):
    def __init__(self, config):
        super(JSDivergenceEvaluation, self).__init__()
        self.config = config
        
        # Initialize variables to store cumulative JSD and SSIM scores
        self.jsd_scores = []
        self.ssim_scores = []
        self.mse_scores = []
        # Initialize variables to store time-stratified JSD and SSIM scores
        self.time_stratified_jsd = {}
        self.time_stratified_ssim = {}
        self.time_stratified_mse = {}
        
    def forward(self, output, padding_masks, all_hash_chunk_indices, gt_probability_map):
        """
        Args:
            output: Tensor containing the model output (B, T, C, H, W)| C = 1
            padding_masks: Tensor containing the padding mask for each sample (B, T)
            all_hash_chunk_indices: (hash_string, idx, total_chunks) for each sample in the batch
            gt_probability_map: Tensor containing the ground truth probability map (T', H, W)
        """
        # Get the corresponding time steps from the ground truth probability map
        gt_probability_map = gt_probability_map[1:output.shape[1]+1]
        
        padding_mask = padding_masks[:, 1:]
        padding_mask = padding_mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        assert output.shape[:2] == padding_mask.shape[:2], f"Shape of outputs and padding masks should be the same {output.shape[:2]} != {padding_mask.shape[:2]}"
        
        # Filter out the batches based on all_hash_chunk_indices (Taking only the first chunk of each hash string)
        batch_indices = [True if x[1] == 0 else False for x in all_hash_chunk_indices]
        output = output[batch_indices]
        padding_mask = padding_mask[batch_indices]  
        
        # Loop over time steps in T axis
        for t in range(output.shape[1]):
            # Get non-padded indices
            non_padded_indices = (padding_mask[:, t] == 1).squeeze()
            if non_padded_indices.numel() == 1:
                non_padded_indices = non_padded_indices.unsqueeze(0)
                
            # Get non-padded outputs and gt_probability_map for the current time step
            output_t = output[non_padded_indices, t]
            gt_map_t = gt_probability_map[t]
            
            # eliminate last element along both axis
            gt_map_t = gt_map_t[:-1, :-1]
            
            assert output_t.shape[2:] == gt_map_t.shape, f"Shape of outputs and gt_probability_map should be the same {output_t.shape[2:]} != {gt_map_t.shape}"

            # Calculate and store JSD if there are non-padded frames at the current time step
            if output_t.numel() > 0 and gt_map_t.numel() > 0:
                self.calculate_and_store_jsd(t, output_t, gt_map_t)
                self.calculate_and_store_ssim_mse(t, output_t, gt_map_t)
    
    def calculate_and_store_jsd(self, t, output_t, gt_map_t):
        # Compute the Jensen-Shannon Divergence for each batch
        jsd_scores_per_batch = []

        for b in range(output_t.shape[0]):
            # Flatten the tensors
            output_b_flat = output_t[b].flatten().cpu().numpy()
            gt_b_flat = gt_map_t.flatten().cpu().numpy()
            assert output_b_flat.shape == gt_b_flat.shape, f"Shape of outputs and gt_probability_map should be the same {output_b_flat.shape} != {gt_b_flat.shape}"
            
            # Compute the JSD
            m = 0.5 * (output_b_flat + gt_b_flat)
            kl_output_m = entropy(output_b_flat, m)
            kl_gt_m = entropy(gt_b_flat, m)
            jsd = 0.5 * kl_output_m + 0.5 * kl_gt_m
            jsd_scores_per_batch.append(jsd)
            
            # print(f"JSD Score for Batch {b}, Time {t}: {jsd}")
            # # Plot output and gt_probability_map side by side
            # plt.figure(figsize=(10, 5))
            # plt.subplot(1, 2, 1)
            # plt.imshow(output_t[b].squeeze().cpu().numpy(), cmap='gray')
            # plt.title('Output')
            # plt.colorbar()
            
            # plt.subplot(1, 2, 2)
            # plt.imshow(gt_map_t.squeeze().cpu().numpy(), cmap='gray')  # Fixed indexing issue here
            # plt.title('Ground Truth')
            # plt.colorbar()
            
            # plt.suptitle(f'Batch {b}, Time {t}')
            # plt.savefig(os.path.join(f'jsd_comparison_batch_.png'))
            # plt.close()
            
            # input("Press Enter to continue...")
            
                
        # Store the average JSD
        avg_jsd = np.mean(jsd_scores_per_batch)
        self.jsd_scores.append(avg_jsd)
        
        # Update time-stratified JSD metrics
        if t not in self.time_stratified_jsd:
            self.time_stratified_jsd[t] = []
        self.time_stratified_jsd[t].append(avg_jsd)
    
    def calculate_and_store_ssim_mse(self, t, output_t, gt_map_t):
        output_t = output_t.squeeze(1) # Remove time dimension
        assert output_t.shape[1:] == gt_map_t.shape, f"Shape of outputs and gt_probability_map should be the same {output_t.shape} != {gt_map_t.shape}"
        ssim_scores_per_batch = []
        mse_scores_per_batch = []

        for b in range(output_t.shape[0]):
            output_b = output_t[b].cpu().numpy()
            gt_b = gt_map_t.cpu().numpy()
            ssim = compare_ssim(output_b, gt_b, win_size=11, data_range=1)
            ssim_scores_per_batch.append(ssim)
            mse_scores_per_batch.append(mean_squared_error(gt_b, output_b))  

        avg_ssim = np.mean(ssim_scores_per_batch)
        self.ssim_scores.append(avg_ssim)
        
        if t not in self.time_stratified_ssim:
            self.time_stratified_ssim[t] = []
        self.time_stratified_ssim[t].append(avg_ssim)

        avg_mse = np.mean(mse_scores_per_batch)
        self.mse_scores.append(avg_mse)
        
        if t not in self.time_stratified_mse:
            self.time_stratified_mse[t] = []
        self.time_stratified_mse[t].append(avg_mse)
        
        
    def get_overall_scores(self):
        overall_scores = {
            "jsd_score": sum(self.jsd_scores) / len(self.jsd_scores) if self.jsd_scores else 0,
            "ssim_score": sum(self.ssim_scores) / len(self.ssim_scores) if self.ssim_scores else 0
        }
        return overall_scores
    
    def get_time_stratified_scores(self):
        return self.time_stratified_jsd, self.time_stratified_ssim
    
    def plot_time_stratified_performance(self, time_stratified_metric_name, save_path=None, start_timestep=0):
        """
        Plots the time-stratified JSD performance.
        """
        if time_stratified_metric_name == "jsd":
            time_stratified_metric = self.time_stratified_jsd
        elif time_stratified_metric_name == "ssim":
            time_stratified_metric = self.time_stratified_ssim
        elif time_stratified_metric_name == "mse":
            time_stratified_metric = self.time_stratified_mse
        
        # Extract time steps and corresponding JSD scores
        time_steps = [t for t in list(time_stratified_metric.keys()) if t >= start_timestep]
        
        # Return if there are no time steps greater than or equal to start_timestep
        if not time_steps:
            print("No data available for timesteps greater than or equal to start_timestep.")
            return
        
        # Calculate average and standard deviation of JSD scores
        avg_scores = [float(np.mean(time_stratified_metric[t])) for t in time_steps]
        std_scores = [float(np.std(time_stratified_metric[t])) for t in time_steps]
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, avg_scores, label='Avg', color='blue', marker='o')
        
        # Add shaded region for standard deviation
        plt.fill_between(time_steps, 
                        np.array(avg_scores) - np.array(std_scores), 
                        np.array(avg_scores) + np.array(std_scores), 
                        color='blue', alpha=0.1)
        
        plt.xlabel('Time Step')
        plt.ylabel(f'{time_stratified_metric_name} Score')
        plt.title(f'Time-Stratified {time_stratified_metric_name} Performance')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout()
        
        save_path_png = os.path.join(save_path, f'time_stratified_{time_stratified_metric_name}_starting_{start_timestep}.png')
        plt.savefig(save_path_png)
        plt.close()
        
        # Save data with standard deviation
        save_path_txt = os.path.join(save_path, f'time_stratified_{time_stratified_metric_name}_starting_{start_timestep}.json')
        data_to_save = {"scale_0": {"time_steps": time_steps, "scores": avg_scores, "std_dev": std_scores}}
        with open(save_path_txt, "w") as txt_file:
            json.dump(data_to_save, txt_file, indent=4)
