import torch
import torch.nn.functional as F


class MultiScaleProcessor:
    @staticmethod
    def downsample_tensor(tensor, scale, segment_flag):
        """
        Downsample a tensor by the given scale.

        Args:
            tensor (torch.Tensor): Input tensor of shape (B, T, C, H, W)
            scale (int): The downsample factor
            segment_flag (bool): Flag to indicate if the input tensor is segmented or BGR

        Returns:
            torch.Tensor: Downsampled tensor
        """
        B, T, C, H, W = tensor.shape
            
        new_H, new_W = H // scale, W // scale

        if segment_flag:
            # If segmented image, use max-pooling
            downsampled = F.max_pool2d(tensor.view(-1, C, H, W), kernel_size=scale).view(B, T, C, new_H, new_W)
        else:
            # If RGB image, use nearest-neighbor downsampling
            downsampled = F.interpolate(tensor.view(-1, C, H, W), size=(new_H, new_W), mode='nearest').view(B, T, C, new_H, new_W)
        
        return downsampled

    @staticmethod
    def get_multiscale_data(collated_batch, scales, segment_flag, verbose=False):
        """
        Create multi-scale data from the given collated_batch.

        Args:
            collated_batch (torch.Tensor): Input tensor of shape (B, T, C, H, W)
            scales (list): List of scales for downsample
            segment_flag (bool): Flag to indicate if the input tensor is segmented or BGR

        Returns:
            list: List of tensors, each of shape (B, T, C, H, W) for different scales
        """
        multi_scale_collated_batch = []
        
        if verbose:
            print(f"\nOriginal batch shape: {collated_batch.shape}")

        for scale in scales:
            downsampled_batch = MultiScaleProcessor.downsample_tensor(collated_batch, scale, segment_flag)
            multi_scale_collated_batch.append(downsampled_batch)
            if verbose:
                print(f"Downsampled batch shape (scale={scale}): {downsampled_batch.shape}")

        return multi_scale_collated_batch


class TestSuite:
    def __init__(self, config):
        self.config = config
        self.trainloader, self.testloader = get_dataloader(config)
    
    def concatenate_gifs_side_by_side(self, batch_idx, scales, channel_idx, segment_flag):
        # List to store the concatenated frames
        concatenated_frames = []

        # Initialize iterators for each scale
        iterators = [ImageSequence.Iterator(Image.open(f'../dump/multi_scale_visualization/Batch_{batch_idx + 1}_Scale_{scale}_{segment_flag}_Channel_{channel_idx}.gif').convert("RGBA")) for scale in scales]

        # Loop through frames
        while True:
            frames = [next(it, None) for it in iterators]
            if None in frames:
                break

            # Concatenate frames side by side
            widths, heights = zip(*(frame.size for frame in frames))
            total_width = sum(widths)
            max_height = max(heights)
            new_frame = Image.new("RGBA", (total_width, max_height))
            x_offset = 0
            for frame in frames:
                new_frame.paste(frame, (x_offset, 0))
                x_offset += frame.width

            concatenated_frames.append(new_frame)

        # Save concatenated GIF
        save_path = f'../dump/multi_scale_visualization/Batch_{batch_idx + 1}_segment_{segment_flag}_Channel_{channel_idx}_Comparison.gif'
        concatenated_frames[0].save(save_path, save_all=True, append_images=concatenated_frames[1:], loop=0, duration=100)
        
    def test_multiscale(self, channel_idx):
        # Settings for multiscale
        scales = [1, 2, 4, 8]
        
        # Process a batch of data
        data, _, _, segmented_data = next(iter(self.trainloader))
        
        multi_scale_data = MultiScaleProcessor.get_multiscale_data(data, scales, segment_flag=False)
        multi_scale_segmented_data = MultiScaleProcessor.get_multiscale_data(segmented_data, scales, segment_flag=True)
        
        # Combining the data into a dictionary
        data_dict = {'original': multi_scale_data, 'segmented': multi_scale_segmented_data}
        
        # Visualize and save the multi-scale data as GIFs
        for data_type, multi_scale_data in data_dict.items():
            for scale_idx, scale_data in enumerate(multi_scale_data):
                scale = scales[scale_idx]
                for batch_idx in range(scale_data.shape[0]):
                    single_batch_data = scale_data[batch_idx].cpu().numpy()
                    
                    # Create directory if not exists and savepath
                    os.makedirs(f'../dump/multi_scale_visualization', exist_ok=True)
                    save_path = f'../dump/multi_scale_visualization/Batch_{batch_idx + 1}_Scale_{scale}_{data_type}_Channel_{channel_idx}.gif'
                    
                    # Save as GIF
                    if data_type == 'segmented':
                        save_gif_from_npy(single_batch_data[:, channel_idx, :, :], save_path)
                    else:
                        save_gif_from_npy(single_batch_data, save_path, format='NCHW')
                        
        # Concatenate GIFs side-by-side
        for batch_idx in range(data.shape[0]):
            for data_type in data_dict.keys():  # Loop over both data types for concatenation
                self.concatenate_gifs_side_by_side(batch_idx, scales, channel_idx, data_type)

                     
if __name__ == "__main__":
    # For debugging
    import sys
    sys.path.append("..")
    import yaml
    from config_utils import load_config, merge_configs, dict_recursive
    from data import get_dataloader
    from visualization_tools import save_gif_from_npy
    import os
    from PIL import Image, ImageSequence
    
    # Configuration Loading
    base_config = load_config('../config/base_config.yaml')
    update_config = load_config('../config/update_config.yaml')
    config = merge_configs(base_config, update_config)
    
    print("Final Configuration:")
    print(yaml.dump(dict_recursive(config)))
    
    # Test multi-scale functionality
    tester = TestSuite(config)
    channel_index_mapping = {"active_fire": 0, "vegetation": 1, "empty": 2, "ember": 3}
    tester.test_multiscale(channel_idx=channel_index_mapping["active_fire"])
