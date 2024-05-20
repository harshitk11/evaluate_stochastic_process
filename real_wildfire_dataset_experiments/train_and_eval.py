from tqdm import tqdm
from torch import nn
import torch
from torch.nn import BCELoss
import os
from utils.visualization_tools import plot_channels_labels_and_output
from utils.evaluation_utils import ScaleWiseEvaluation
import numpy as np
import torch.nn.functional as F
from utils.stochasticity_estimator_nextdaywildfire import calculate_similarity_metrics
import json


class ScaleWiseLoss(nn.Module):
    def __init__(self, scales, learn_weights, args):
        super(ScaleWiseLoss, self).__init__()
        self.args = args
        self.scales = scales

        if args.experiment_setting.train.multiscale.custom.use:
            custom_weights = args.experiment_setting.train.multiscale.custom.weights
            assert isinstance(custom_weights, list), "Custom weights must be a list"
            assert len(custom_weights) == len(scales), "Length of custom weights must match the number of scales"
            assert all(isinstance(weight, (int, float)) for weight in custom_weights), "All custom weights must be numbers"
            print(f"Using custom weights: {custom_weights}")
            self.scale_loss_weights = nn.Parameter(torch.tensor(custom_weights, requires_grad=learn_weights), requires_grad=learn_weights)
        elif learn_weights:
            self.scale_loss_weights = nn.Parameter(torch.ones(len(scales), requires_grad=True))
        else:
            self.scale_loss_weights = torch.ones(len(scales))

    def get_trained_weights(self):
        return self.scale_loss_weights.detach().cpu().numpy()

    def forward(self, outputs, targets, padding_masks, loss_calculation_scales=None, scale_loss_flag=True):
        """
        Args:
            - outputs (list of torch.Tensor): A list of tensors for different scales, each tensor having a shape of (B, T, C, H, W)
            - targets (list of torch.Tensor): A list of tensors for different scales, each tensor having a shape of (B, T, C, H, W)
            - padding_masks (torch.Tensor): A tensor of shape (B, T) containing the padding masks for each batch
        
        Returns:
            - total_loss (torch.Tensor): A scalar tensor containing the total loss across all scales
            - scale_loss_dict (dict): A dictionary containing the loss for each scale
        """
        loss_calculation_scales = loss_calculation_scales if loss_calculation_scales else self.scales
        scale_loss_dict = {}
        
        loss_fn = nn.BCELoss(reduction='none')  #loss per frame
        total_loss = 0

        for scale_idx, scale in enumerate(self.scales):
            if scale in loss_calculation_scales:
                output = outputs[scale_idx]
                target = targets[scale_idx]

                assert output.shape == target.shape, f"Shape of outputs and targets should be the same across scales {output.shape} != {target.shape}"
                scale_loss = loss_fn(output, target)
            
                # Apply padding mask
                padding_mask = padding_masks[:, 1:]  # Assuming padding masks need to be adjusted for target frames
                padding_mask = padding_mask.unsqueeze(2).unsqueeze(3).unsqueeze(4)  # Adjusting shape to match the output and target
                assert output.shape[:2] == padding_mask.shape[:2], f"Shape of outputs and padding masks should be the same across scales {output.shape[:2]} != {padding_mask.shape[:2]}"
                scale_loss *= padding_mask
            
                # Average loss per pixel per non-padded frame
                scale_loss = (scale_loss.sum() / (output.shape[3] * output.shape[4])) / padding_mask.sum()
                # Scale loss by the weight for this scale if scale_loss_flag is True
                weighted_scale_loss = self.scale_loss_weights[scale_idx] * scale_loss if scale_loss_flag else scale_loss
                
                # print(f"Scale: {scale}, Weight: {self.scale_loss_weights[scale_idx]}, Loss: {scale_loss}, Weighted Loss: {weighted_scale_loss}")
                scale_loss_dict[scale] = weighted_scale_loss
                
                total_loss += weighted_scale_loss
        
        return total_loss, scale_loss_dict

def calculate_weighted_bce_loss(raw_output, labels, weight_for_minority=10.0):
    """
    Calculate weighted BCE loss, considering that labels can have a value of -1 which should be ignored.
    
    Parameters:
    - raw_output : torch.Tensor
        The raw output from the model, shape = (B, 1, 1, H, W)
    - labels : torch.Tensor
        Ground truth labels, shape = (B, 1, 1, H, W)
    - weight_for_minority : float
        The weight to assign to the minority class
    
    Returns:
    - bce_loss : torch.Tensor
        The calculated weighted BCE loss
    """
    
    # Create a mask to ignore the unlabeled data in the loss calculation
    mask = (labels != -1)
    
    # Create weights tensor, assigning higher weights to the minority class
    weights = torch.ones_like(labels).float() * weight_for_minority
    weights[labels == 0] = 1.0  # Majority class weight is set to 1
    
    # Apply masks to outputs, labels, and weights
    output_masked = raw_output[mask]
    labels_masked = labels[mask]
    weights_masked = weights[mask]
    
    # Calculate BCE loss with the weights
    bce_loss = F.binary_cross_entropy(output_masked, labels_masked.float(), weight=weights_masked)
    
    return bce_loss

class Trainer:
    segmented_channel_lut = {"active_fire": 0, "vegetation": 1, "empty": 2, "ember": 3}
        
    def __init__(self, config, model, optimizer, trainloader, testloader, writer=None):
        # Config Variables
        self.config = config
        self.DEVICE = config.dataloader.device
        
        self.TRAIN_MULTISCALE_USE = config.experiment_setting.train.multiscale.use
        self.TRAIN_SCALES = config.experiment_setting.train.multiscale.scales
        self.TRAIN_LOSS_CALCULATION_SCALES = config.experiment_setting.train.multiscale.loss_calculation_scales
        
        self.EVAL_MULTISCALE_USE = config.experiment_setting.evaluation.multiscale.use
        self.EVAL_SCALES = config.experiment_setting.evaluation.multiscale.scales
        self.EVAL_LOSS_CALCULATION_SCALES = config.experiment_setting.evaluation.multiscale.loss_calculation_scales
        self.EVAL_ONLY = config.experiment_setting.evaluation.eval_only
        
        self.NUM_OBSERVED_FRAMES = config.experiment_setting.chunk_params.num_observed
        self.NUM_PREDICTED_FRAMES = config.experiment_setting.chunk_params.num_predicted
        assert self.NUM_OBSERVED_FRAMES+self.NUM_PREDICTED_FRAMES == config.dataloader.chunk_params.chunk_size, "Number of observed frames and number of predicted frames must sum to total number of frames"
        
        # self.FILTER_INPUTS = config.experiment_setting.input_filtering.use
        # self.DC_THRESHOLD = config.experiment_setting.input_filtering.dc_threshold
        
        self.model = model
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.testloader = testloader
        self.writer = writer
        
        self.weight_for_minority = config.experiment_setting.train.weight_for_minority_class
        
        self.multi_scale_processor = MultiScaleProcessor()
        self.LEARN_MULTISCALE_WEIGHTS = config.experiment_setting.train.multiscale.learned_multiscale_weights
        self.scalewise_loss = ScaleWiseLoss(self.TRAIN_SCALES, learn_weights=self.LEARN_MULTISCALE_WEIGHTS, args=config)
        if self.LEARN_MULTISCALE_WEIGHTS:
            self.optimizer.add_param_group({'params': self.scalewise_loss.parameters()})

        # Evaluator
        self.evaluator = ScaleWiseEvaluation(scales=self.EVAL_SCALES, config=config)
        
        # Stochasticity Estimator
        self.stochasticity_estimator_dict = {}

    def train_step(self, epoch):
        self.model.train()
        train_loss = 0.0
        num_chunks = 0
        num_filtered_samples = 0
        total_samples = 0

        # Progress bar (optional)
        pbar = tqdm(self.trainloader, desc=f"Training Epoch {epoch}")

        for batch_idx, (inputs, labels) in enumerate(pbar):
            """
            Data shapes:
             - collated_batch: (B, T, C, H, W)
             - segmented_collated_batch: (B, T, C, H, W)
             - padding_masks: (B, T) 
             - all_hash_chunk_indices: (B)
            """
            # Convert TensorFlow tensors to numpy arrays
            inputs_np = inputs.numpy()
            labels_np = labels.numpy()
            
            # Convert numpy arrays to PyTorch tensors
            inputs_torch = torch.tensor(inputs_np, device=self.DEVICE)  
            labels_torch = torch.tensor(labels_np, device=self.DEVICE)  
            
            # Convert shape of inputs_torch from (B,H,W,C) to (B,T,C,H,W)
            inputs_torch = inputs_torch.permute(0, 3, 1, 2).unsqueeze(1) # (B, 1, 12, H, W)
            labels_torch = labels_torch.permute(0, 3, 1, 2).unsqueeze(1) # (B, 1, 1, H, W)
            # print(f"Input shape: {inputs_torch.shape}, Label shape: {labels_torch.shape}")
            total_samples += inputs_torch.shape[0]
                     
            # Forward Pass | output_ms_list: [(B, T, C, H1, W1), (B, T, C, H2, W2), ...] of length self.TRAIN_SCALES
            model_output = self.model(inputs_torch) 
            
            # ###################### Visualization ######################
            # experiment_name = self.config.experiment_setting.evaluation.model_weights_path.split("/")[-1]
            # viz_dir = os.path.join("./dump", "visualization_dataloader", experiment_name)
            # os.makedirs(viz_dir, exist_ok=True)
            # scale_map = {1: 0, 2: 1, 4: 2, 8: 3}
            # for scale, scale_indx in scale_map.items():
            #     # scale = 8
            #     # scale_indx = scale_map[scale]
            #     visualize_frames(bgr_gt = collated_batch[:,1:,:,:,:], 
            #                     segmented_gt = target_segmented[scale_indx], 
            #                     prediction = output_ms_list[scale_indx], 
            #                     channel_idx=0, 
            #                     save_path=os.path.join(viz_dir, f'{experiment_name}_visualization_batch_{batch_idx}_scale_{scale}.png'),
            #                     #  save_path=os.path.join(viz_dir, f'visualization_batch.png'),
            #                     skip_frames=5)
            # input("Press Enter to continue...")
            # continue
            # ##########################################################
            
            # Calculate loss
            total_loss = calculate_weighted_bce_loss(model_output[0], 
                                                     labels_torch, 
                                                     weight_for_minority=self.weight_for_minority)
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Update training loss (Average loss per non-padded frame)
            train_loss += total_loss.item()
            num_chunks += inputs_torch.shape[0]
            
            # Update progress bar
            pbar.set_postfix({'Training Loss': train_loss / num_chunks, "Filtered Samples": num_filtered_samples})


            # if batch_idx == 0:
            #     break
                
        # Compute average loss
        avg_train_loss = train_loss / num_chunks
        
        return {'train_loss': avg_train_loss, "num_filtered_samples": num_filtered_samples, "total_samples": total_samples}

    def eval_step(self, epoch):
        verbose = False
        self.model.eval()
        val_loss = 0.0
        num_samples = 0
        gt_probability_map = None
        num_filtered_samples = 0

        # Progress bar (optional)
        pbar = tqdm(self.testloader, desc=f"Evaluation Epoch {epoch}")

        with torch.no_grad():
            for batch_idx, (inputs, labels) in enumerate(pbar):
                """
                Data shapes:
                - inputs: (B, T, C, H, W)
                - labels: (B, T, C, H, W)
                """

                # Convert TensorFlow tensors to numpy arrays
                inputs_np = inputs.numpy()
                labels_np = labels.numpy()

                # Convert numpy arrays to PyTorch tensors
                inputs_torch = torch.tensor(inputs_np, device=self.DEVICE)  
                labels_torch = torch.tensor(labels_np, device=self.DEVICE)  

                # Convert shape of inputs_torch from (B,H,W,C) to (B,T,C,H,W)
                inputs_torch = inputs_torch.permute(0, 3, 1, 2).unsqueeze(1) # (B, 1, C, H, W)
                labels_torch = labels_torch.permute(0, 3, 1, 2).unsqueeze(1) # (B, 1, C, H, W)
                
            
                # Forward Pass
                model_output = self.model(inputs_torch)
                 
                stochasticity_estimate = None
                if self.EVAL_ONLY:
                    # Load the corresponding ground truth probability map if sameInit dataset is used 
                    eval_dataset_name = self.config.experiment_setting.evaluation.dataset_name
                    
                    all_hash_chunk_indices = [("hash_string", 0, 1)]
                    padding_masks = torch.ones((inputs_torch.shape[0], inputs_torch.shape[1])) 
                    output_ms_list = [model_output[0]]
                    target_segmented = [labels_torch]
                    
                    
                    optimal_threshold = self.evaluator(outputs=output_ms_list,
                                targets=target_segmented,
                                padding_masks=padding_masks,
                                threshold=0.5,
                                loss_calculation_scales=[1],
                                all_hash_chunk_indices=all_hash_chunk_indices,
                                sample_index=batch_idx)
                
                    # Calculate the stochasticity estimation
                    input_plot = inputs_torch.permute(0,3,4,2,1).squeeze(-1).cpu().numpy()
                    label_plot = labels_torch.permute(0,3,4,2,1).squeeze(-1).cpu().numpy()
                    stochasticity_estimate = calculate_similarity_metrics(input_plot, label_plot)
                    self.stochasticity_estimator_dict[batch_idx] = stochasticity_estimate
                    
                # visualize = False
                # if stochasticity_estimate['Dice_Similarity'] > 0.0 and stochasticity_estimate['Dice_Similarity'] < 0.1:
                #     visualize = True
                # if visualize:
                #     ###################### Visualization ######################
                #     experiment_name = self.config.experiment_setting.experiment_name
                #     # viz_dir = os.path.join("./dump", "prediction_visualizations", experiment_name, str(batch_idx))
                #     viz_dir = os.path.join("./dump", "prediction_visualizations", experiment_name)
                #     os.makedirs(viz_dir, exist_ok=True)
                #     viz_path = os.path.join(viz_dir, f'{batch_idx}_{stochasticity_estimate["Dice_Similarity"]}_{experiment_name}_visualization_batch.pdf')
                    
                #     input_plot = inputs_torch.permute(0,3,4,2,1).squeeze(-1).cpu().numpy()
                #     label_plot = labels_torch.permute(0,3,4,2,1).squeeze(-1).cpu().numpy()
                #     output_plot = model_output[0].permute(0,3,4,2,1).squeeze(-1).cpu().numpy()
                #     performance_score = {"auc_pr": self.evaluator.auc_pr_scores[1][-1], 
                #                          "mse": self.evaluator.mse_scores[1][-1], 
                #                          "recall": self.evaluator.recall_scores[1][-1], 
                #                          "precision": self.evaluator.precision_scores[1][-1]}
                #     plot_channels_labels_and_output(input_plot, label_plot, output_plot, 
                #                                     batch_index=0, 
                #                                     save_path=viz_path, 
                #                                     only_prev_fire_mask=True, 
                #                                     performance_score = performance_score,
                #                                     stochasticity_estimate=stochasticity_estimate)
                #     input("Press Enter to continue...")
                #     continue
                #     ##########################################################
                
                # Calculate loss
                total_loss = calculate_weighted_bce_loss(model_output[0], 
                                                         labels_torch,
                                                         weight_for_minority=self.weight_for_minority)
                
                # Update validation loss
                val_loss += total_loss.item()
                num_samples += inputs_torch.shape[0]
                
                # Update progress bar
                pbar.set_postfix({'Validation Loss': val_loss / num_samples, "Filtered Samples": num_filtered_samples})
                
                # if batch_idx == 0:
                #     break

        # Compute average loss
        avg_val_loss = val_loss / num_samples
        
        evaluation_scores = None
        if self.EVAL_ONLY:
            evaluation_scores = self.evaluator.get_overall_scores()
            for metric, metric_scores in evaluation_scores.items():
                for scale, scale_scores in metric_scores.items():
                    print(f"Scale: {scale}, Metric: {metric}, Scores: {scale_scores}")
                    
            # Save raw evaluation scores
            experiment_name = self.config.experiment_setting.experiment_name
            dirPath = os.path.join("./dump", "raw_evaluation_scores", experiment_name)
            os.makedirs(dirPath, exist_ok=True)
            self.evaluator.save_raw_scores(savepath=dirPath)
            
            # Save stochasticity estimates
            stoch_est_path = os.path.join("./dump", "stochasticity_estimates", experiment_name)
            os.makedirs(stoch_est_path, exist_ok=True)
            filepath_stoch = os.path.join(stoch_est_path, 'stochasticity_estimates.json')
            def json_serializable(item):
                if isinstance(item, np.float32):
                    return float(item)
                raise TypeError(f"Type {type(item)} not serializable")
            with open(filepath_stoch, 'w', encoding='utf-8') as file:
                json.dump(self.stochasticity_estimator_dict, file, default=json_serializable, ensure_ascii=False, indent=4)
                
                
        return {'val_loss': avg_val_loss, "evaluation_scores": evaluation_scores}


