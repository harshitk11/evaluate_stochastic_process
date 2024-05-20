from tqdm import tqdm
from torch import nn
import torch
from utils.multiscale import MultiScaleProcessor
from torch.nn import BCELoss
import os
from torchvision.utils import save_image
from utils.visualization_tools import visualize_frames, visualize_frames_different_thresholds
from utils.evaluation_utils import ScaleWiseEvaluation
from utils.evaluation_distribution_utils import JSDivergenceEvaluation
from utils.distribution_visualization import visualize_distribution_predictions
import numpy as np
import pickle


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
        
        self.model = model
        self.optimizer = optimizer
        self.trainloader = trainloader
        self.testloader = testloader
        self.writer = writer
        
        self.multi_scale_processor = MultiScaleProcessor()
        self.LEARN_MULTISCALE_WEIGHTS = config.experiment_setting.train.multiscale.learned_multiscale_weights
        self.scalewise_loss = ScaleWiseLoss(self.TRAIN_SCALES, learn_weights=self.LEARN_MULTISCALE_WEIGHTS, args=config)
        if self.LEARN_MULTISCALE_WEIGHTS:
            self.optimizer.add_param_group({'params': self.scalewise_loss.parameters()})

        # Evaluator
        self.evaluator = ScaleWiseEvaluation(scales=self.EVAL_SCALES, config=config)
        self.distribution_evaluator = JSDivergenceEvaluation(config=config)
        
        # prediction, forecastGT tracker dict
        self.prediction_forecastGT_tracker = {}
        
    def extract_prediction_forecastGT(self, output, gt_probability_map, observed_GT, padding_masks, all_hash_chunk_indices, sample_idx):
        """
        Args:
            output: Tensor containing the model output (B, T, C, H, W)| C = 1
            gt_probability_map: Tensor containing the ground truth probability map (T', H, W)
            observed_GT: Tensor containing the observed ground truth (B, T, C, H, W) | C = 1
            padding_masks: Tensor containing the padding mask for each sample (B, T)
            all_hash_chunk_indices: (hash_string, idx, total_chunks) for each sample in the batch
            sample_idx: Index of the sample in the dataset
        """
        if gt_probability_map is not None:
            # Get the corresponding time steps from the ground truth probability map and padding masks
            gt_probability_map = gt_probability_map[1:output.shape[1]+1]
        padding_mask = padding_masks[:, 1:]
        
        # Filter out the batches based on all_hash_chunk_indices (Taking only the first chunk of each hash string)
        batch_indices = [True if x[1] == 0 else False for x in all_hash_chunk_indices]
        output = output[batch_indices]
        padding_mask = padding_mask[batch_indices]
        observed_GT = observed_GT[batch_indices]
        
        # Filter out the non-padded frames
        non_padded_indices = (padding_mask == 1).squeeze()
        output = output.squeeze()[non_padded_indices]
        observed_GT = observed_GT.squeeze()[non_padded_indices]
        if gt_probability_map is not None:
            gt_probability_map = gt_probability_map[non_padded_indices]
            gt_probability_map = gt_probability_map[:,:-1, :-1]
            
            assert output.shape == gt_probability_map.shape, f"Shape of outputs and gt_probability_map should be the same {output.shape} != {gt_probability_map.shape}"
        assert output.shape == observed_GT.shape, f"Shape of outputs and observed_GT should be the same {output.shape} != {observed_GT.shape}"
        
        # Convert output and gt_probability_map to numpy
        output = output.detach().cpu().numpy()
        if gt_probability_map is not None:
            gt_probability_map = gt_probability_map.detach().cpu().numpy()
        observed_GT = observed_GT.detach().cpu().numpy()
        
        # update the tracker dict
        self.prediction_forecastGT_tracker[sample_idx] = {"prediction": output, "forecastGT": gt_probability_map, "observedGT": observed_GT}
        
    def dump_prediction_forecastGT(self, fpath):
        """
        Args:
            fpath: path to save the tracker dict
        """
        # Save the tracker dict as a pickle file
        with open(fpath, 'wb') as f:
            pickle.dump(self.prediction_forecastGT_tracker, f)
            
        
    def prepare_chunks(self, collated_batch_multiscale, verbose=False):
        """
        Divides each tensor in the multi-scale data into observations and targets.

        Args:
            collated_batch_multiscale (list): List of tensors, each of shape (B, T, C, H, W)
                                            for different scales
            verbose (bool): Flag to indicate if shapes should be printed for debugging

        Returns:
            obs (list): List of tensors containing the first NUM_OBSERVED_FRAMES for each scale
            target (list): List of tensors containing the next NUM_PREDICTED_FRAMES for each scale
        """
        obs = []
        target = []
        
        # For each scale in collated_batch_multiscale
        for collated_batch in collated_batch_multiscale:
            B, T, C, H, W = collated_batch.shape
            
            # Make sure the total number of frames T matches with NUM_OBSERVED_FRAMES + NUM_PREDICTED_FRAMES
            assert T == self.NUM_OBSERVED_FRAMES + self.NUM_PREDICTED_FRAMES, "Total frames should be equal to NUM_OBSERVED_FRAMES + NUM_PREDICTED_FRAMES"
            
            # Separate the observation and target frames for this scale
            obs_frames = collated_batch[:, :self.NUM_OBSERVED_FRAMES, :, :, :]
            target_frames = collated_batch[:, self.NUM_OBSERVED_FRAMES:, :, :, :]
            
            if verbose:
                # Print shapes for debugging
                print(" - Chunk Preparation")
                print(f"Observation frames shape: {obs_frames.shape}")
                print(f"Target frames shape: {target_frames.shape}")
                
            # Trim H,W from 65,65 to 64,64
            obs_frames = obs_frames[:, :, :, :-1, :-1]
            target_frames = target_frames[:, :, :, :-1, :-1]
            
            obs.append(obs_frames)
            target.append(target_frames)
            
        return obs, target

    def prepare_target(self, segmented_collated_batch_multiscale, clip_first_frame=None, segmented_channel=None, verbose=False):
        """
        Prepares the target tensors by possibly clipping the first frame and selecting specific channels.

        Args:
            - segmented_collated_batch_multiscale (list of torch.Tensor): A list of tensors for different scales, each tensor having a shape of (B, T, C, H, W)
            - clip_first_frame (bool, optional): If True, the first frame from each batch and scale will be skipped. Defaults to None.
            - segmented_channel (str or list of str, optional): The channel(s) to be used for segmentation. Can be a string or a list of strings representing multiple channels. 
                                                                If it is a list, the corresponding channels are extracted and the element-wise addition of the channels is performed. 
                                                                It can be one of the following: "active_fire", "vegetation", "empty", "ember". Defaults to None.
            - verbose (bool, optional): If True, enables verbose mode for debugging. Defaults to False.

        Returns:
            - target_segmented (list of torch.Tensor): A list of target tensors prepared according to the specified clip_first_frame and segmented_channel options, retaining the shape of the input tensors.

        """
        if verbose:
            print(" - Target preparation:")
            print(f"Input shapes: {[batch.shape for batch in segmented_collated_batch_multiscale]}")
        
        # Trim H,W from 65,65 to 64,64
        segmented_collated_batch_multiscale = [segmented_collated_batch[:, :, :, :-1, :-1] for segmented_collated_batch in segmented_collated_batch_multiscale]
        
        if clip_first_frame:
            # Skip the first frame for each batch and scale
            segmented_collated_batch_multiscale = [segmented_collated_batch[:, 1:, :, :, :] for segmented_collated_batch in segmented_collated_batch_multiscale]
        
        if segmented_channel:
            if isinstance(segmented_channel, list):
                # Get the segmented channels and perform element-wise addition
                channel_indices = [self.segmented_channel_lut[channel] for channel in segmented_channel]
                segmented_collated_batch_multiscale = [
                    torch.clamp(    # For downsampled images, the sum of the channels can exceed 1.0. Clamp to 1.0
                        sum(segmented_collated_batch[:, :, idx:idx+1, :, :] for idx in channel_indices),  
                        max=1
                    ) 
                    for segmented_collated_batch in segmented_collated_batch_multiscale
                ]    
            else:
                raise ValueError("Expected list")
            
        if verbose:
            print(f"Output shapes: {[batch.shape for batch in segmented_collated_batch_multiscale]}")

            # Create directory if not exists
            os.makedirs('./dump/target_preparation/', exist_ok=True)
            
            # Convert segmented_channel to a string representation for the filename
            if isinstance(segmented_channel, list):
                segmented_channel_str = "_".join(segmented_channel)
            else:
                segmented_channel_str = segmented_channel

            # Save the visualization for one batch across different scales
            for scale_idx, scale_batch in enumerate(segmented_collated_batch_multiscale):
                for batch_idx, batch in enumerate(scale_batch):
                    # Save the first channel of the first time frame of each batch
                    save_image(batch[0, 0], f'./dump/target_preparation/batch_{batch_idx}_channel_{segmented_channel_str}_scale_{scale_idx}.png')

        return segmented_collated_batch_multiscale


    def train_step(self, epoch):
        verbose = False
        self.model.train()
        train_loss = 0.0
        num_chunks = 0

        # Progress bar (optional)
        pbar = tqdm(self.trainloader, desc=f"Training Epoch {epoch}")

        for batch_idx, (collated_batch, padding_masks, all_hash_chunk_indices, segmented_collated_batch) in enumerate(pbar):
            """
            Data shapes:
             - collated_batch: (B, T, C, H, W)
             - segmented_collated_batch: (B, T, C, H, W)
             - padding_masks: (B, T) 
             - all_hash_chunk_indices: (B)
            """
            collated_batch = collated_batch.to(self.DEVICE)
            segmented_collated_batch = segmented_collated_batch.to(self.DEVICE)
            padding_masks = padding_masks.to(self.DEVICE)
            
            # Multiscale data preparation
            if self.TRAIN_MULTISCALE_USE and self.TRAIN_SCALES is not None:
                collated_batch_multiscale = self.multi_scale_processor.get_multiscale_data(collated_batch, self.TRAIN_SCALES, segment_flag=False, verbose=verbose)
                segmented_collated_batch_multiscale = self.multi_scale_processor.get_multiscale_data(segmented_collated_batch, self.TRAIN_SCALES, segment_flag=True, verbose=verbose)            
                
            # observations bgr chunks 
            obs_bgr, pred_bgr = self.prepare_chunks(collated_batch_multiscale, verbose=verbose)
            
            # Prepare target | target_segmented: [(B, T, C, H1, W1), (B, T, C, H2, W2), ...] of length self.TRAIN_SCALES
            target_segmented = self.prepare_target(segmented_collated_batch_multiscale,
                                                    clip_first_frame=True,
                                                    segmented_channel=["ember", "active_fire"],     # segmented_channel options: "active_fire", "vegetation", "empty", "ember"
                                                    verbose=verbose)
            
            # Forward Pass | output_ms_list: [(B, T, C, H1, W1), (B, T, C, H2, W2), ...] of length self.TRAIN_SCALES
            output_ms_list = self.model(obs_bgr[0]) 
            
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
            total_loss, _ = self.scalewise_loss(outputs = output_ms_list, 
                                             targets = target_segmented, 
                                             padding_masks = padding_masks,
                                             loss_calculation_scales = self.TRAIN_LOSS_CALCULATION_SCALES,
                                             scale_loss_flag = True)
            
            # for arnca and stochasticity 80
            if torch.isnan(total_loss):
                print(f"Skipping batch {batch_idx} due to NaN loss.")
                continue
            
            # print(self.optimizer.param_groups)
            # print(self.scalewise_loss.scale_loss_weights.grad)
            # exit()
            # Backward pass and optimization
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Update training loss (Average loss per non-padded frame)
            train_loss += total_loss.item()
            num_chunks += len(collated_batch)
            
            # Update progress bar
            pbar.set_postfix({'Training Loss': train_loss / num_chunks})

            # Update progress bar 
            pbar.set_postfix({"Loss": train_loss / (batch_idx + 1)})

            # if batch_idx == 0:
            #     break
                
        # Compute average loss
        avg_train_loss = train_loss / num_chunks

        # Learned multiscale weights for logging
        detached_weights = None
        if self.LEARN_MULTISCALE_WEIGHTS:
            detached_weights = self.scalewise_loss.get_trained_weights()
        
        return {'train_loss': avg_train_loss, 'learned_multiscale_weights': detached_weights}

    def eval_step(self, epoch):
        verbose = False
        self.model.eval()
        val_loss = 0.0
        num_samples = 0
        gt_probability_map = None

        # Progress bar (optional)
        pbar = tqdm(self.testloader, desc=f"Evaluation Epoch {epoch}")

        with torch.no_grad():
            for batch_idx, (collated_batch, padding_masks, all_hash_chunk_indices, segmented_collated_batch) in enumerate(pbar):
                """
                Data shapes:
                - collated_batch: (B, T, C, H, W)
                - segmented_collated_batch: (B, T, C, H, W)
                - padding_masks: (B, T)
                - all_hash_chunk_indices: (B)
                """
                collated_batch = collated_batch.to(self.DEVICE)
                segmented_collated_batch = segmented_collated_batch.to(self.DEVICE)
                padding_masks = padding_masks.to(self.DEVICE)
                
                # Multiscale data preparation
                if self.EVAL_MULTISCALE_USE and self.EVAL_SCALES is not None:
                    collated_batch_multiscale = self.multi_scale_processor.get_multiscale_data(collated_batch, self.EVAL_SCALES, segment_flag=False, verbose=verbose)
                    segmented_collated_batch_multiscale = self.multi_scale_processor.get_multiscale_data(segmented_collated_batch, self.EVAL_SCALES, segment_flag=True, verbose=verbose)
                
                # Observations BGR chunks
                obs_bgr, pred_bgr = self.prepare_chunks(collated_batch_multiscale, verbose=verbose)
                
                # Prepare target | target_segmented: [(B, T, C, H1, W1), (B, T, C, H2, W2), ...] of length self.EVAL_SCALES
                target_segmented = self.prepare_target(segmented_collated_batch_multiscale, 
                                                    clip_first_frame=True, 
                                                    segmented_channel=["ember", "active_fire"], 
                                                    verbose=verbose)
                
                # Forward Pass | output_ms_list: [(B, T, C, H1, W1), (B, T, C, H2, W2), ...] of length self.EVAL_SCALES
                output_ms_list = self.model(obs_bgr[0]) 
                # NOTE: Do one lstm per scale
                
                
                # Calculate loss
                total_loss, _ = self.scalewise_loss(outputs=output_ms_list, 
                                                targets=target_segmented, 
                                                padding_masks=padding_masks,
                                                loss_calculation_scales=self.EVAL_LOSS_CALCULATION_SCALES,
                                                scale_loss_flag=False)
                
                if self.EVAL_ONLY:
                    # Load the corresponding ground truth probability map if sameInit dataset is used 
                    eval_dataset_name = self.config.experiment_setting.evaluation.dataset_name
                    
                    if "sameInit" in eval_dataset_name:
                        if gt_probability_map is None:
                            # Load the corresponding ground truth probability map
                            print(f"Loading ground truth probability map for {eval_dataset_name}")
                            probability_map_path = os.path.join("./dump/probabilityMap_MC", eval_dataset_name, eval_dataset_name + ".npy")
                            if not os.path.exists(probability_map_path):
                                raise ValueError(f"Probability map not found at {probability_map_path}")
                            # Shape: (T, H, W)
                            gt_probability_map = torch.from_numpy(np.load(probability_map_path)).to(self.DEVICE)    
                        
                        # Compute JSD
                        self.distribution_evaluator(output = output_ms_list[0], 
                                                    padding_masks = padding_masks, 
                                                    all_hash_chunk_indices = all_hash_chunk_indices, 
                                                    gt_probability_map = gt_probability_map)
                        
                    # Track prediction,forecastGT, and observedGT for reliability plot analysis and sampling study
                    self.extract_prediction_forecastGT(output = output_ms_list[0],
                                                        gt_probability_map = gt_probability_map,
                                                        observed_GT = target_segmented[0],
                                                        padding_masks = padding_masks,
                                                        all_hash_chunk_indices = all_hash_chunk_indices,
                                                        sample_idx = batch_idx)
                    
                    optimal_threshold = self.evaluator(outputs=output_ms_list,
                                targets=target_segmented,
                                padding_masks=padding_masks,
                                threshold=0.5,
                                loss_calculation_scales=self.EVAL_LOSS_CALCULATION_SCALES,
                                all_hash_chunk_indices=all_hash_chunk_indices,
                                gt_probability_map=gt_probability_map)
                
                # ###################### Visualization ######################
                # experiment_name = self.config.experiment_setting.experiment_name
                # viz_dir = os.path.join("./dump", "visualization_dataloader", experiment_name, str(batch_idx))
                # os.makedirs(viz_dir, exist_ok=True)
                # scale_map = {1: 0, 2: 1, 4: 2, 8: 3}
                # for scale, scale_indx in scale_map.items():
                #     if scale in [1]:
                #         # scale = 8
                #         # scale_indx = scale_map[scale]
                #         visualize_frames_different_thresholds(bgr_gt = collated_batch[:,1:,:,:,:], 
                #                         segmented_gt = target_segmented[scale_indx], 
                #                         prediction = output_ms_list[scale_indx], 
                #                         channel_idx=0, 
                #                         save_path=os.path.join(viz_dir, f'visualization_B_{batch_idx}_{experiment_name}_scale_{scale}.png'),
                #                         skip_frames=5,
                #                         gt_probability_map=gt_probability_map,
                #                         optimal_threshold=optimal_threshold[scale])
                # # input("Press Enter to continue...")
                
                # experiment_name = self.config.experiment_setting.experiment_name
                # viz_dir = os.path.join("./dump", "visualization_dataloader", experiment_name, str(batch_idx))
                # os.makedirs(viz_dir, exist_ok=True)
                # scale_map = {1: 0, 2: 1, 4: 2, 8: 3}
                # for scale, scale_indx in scale_map.items():
                #     if scale in [1]:
                #         # scale = 8
                #         # scale_indx = scale_map[scale]
                #         visualize_distribution_predictions(bgr_gt = collated_batch[:,1:,:,:,:], 
                #                         prediction = output_ms_list[scale_indx], 
                #                         save_path=os.path.join(viz_dir, f'distribution_B_{batch_idx}_{experiment_name}_scale_{scale}'),
                #                         skip_frames=5)
                # input("Press Enter to continue...")
                # continue
                # # ##########################################################
                
                # Update validation loss (Average loss per non-padded frame)
                
                # check if total_loss.item() loss is NaN, if yes then skip
                if np.isnan(total_loss.item()):
                    print(f"Validation loss is NaN: {total_loss.item()}")
                else:
                    val_loss += total_loss.item()
                    num_samples += len(collated_batch)
                
                # Update progress bar
                pbar.set_postfix({'Validation Loss': val_loss / num_samples})
                if np.isnan(val_loss / num_samples):
                    print(f"Validation loss is NaN: {total_loss.item(), num_samples}")
                
                # if batch_idx == 10:
                #     break
                
            avg_val_loss = val_loss / num_samples
            
            if self.EVAL_ONLY:
                experiment_name = self.config.experiment_setting.experiment_name
                eval_dir = os.path.join("./dump", "evaluator", experiment_name)
                os.makedirs(eval_dir, exist_ok=True)
                self.evaluator.plot_time_stratified_performance(save_path=eval_dir)
            
                if "sameInit" in eval_dataset_name:
                    self.distribution_evaluator.plot_time_stratified_performance(time_stratified_metric_name="jsd", save_path=eval_dir, start_timestep=10)
                    self.distribution_evaluator.plot_time_stratified_performance(time_stratified_metric_name="ssim", save_path=eval_dir, start_timestep=10)
                    self.distribution_evaluator.plot_time_stratified_performance(time_stratified_metric_name="mse", save_path=eval_dir, start_timestep=10)

                # Dump prediction and forecastGT for reliability plot analysis
                dump_dir = os.path.join("./dump", "prediction_forecastGT_raw_data", experiment_name)
                os.makedirs(dump_dir, exist_ok=True)
                self.dump_prediction_forecastGT(fpath=os.path.join(dump_dir, f"raw_prediction_forecastGT.pkl"))
                    
            return {'val_loss': avg_val_loss} 
        

