import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random
from utils.video_parser import png_to_npy, Segmenter, MAX_PIXEL_VALUE
from utils.config_utils import load_config, merge_configs, save_config, dict_recursive
from utils.visualization_tools import save_gif_from_npy
import yaml
import re


class ForestFireDataset(Dataset):
    TRAIN_DIR = 'Train'
    TEST_DIR = 'Test'
    
    def __init__(self, args, root_directory, split):
        """
        Args:
            args: Configuration arguments.
            root_directory (string): Root directory containing data.
            split (string): Either 'Train' or 'Test' indicating which data split to use.
        """
        # Initialize arguments
        train_val_split = args.dataloader.train_val_split
        seed = args.dataloader.seed
        
        # Check if .npy files are already parsed
        train_path = os.path.join(root_directory, self.TRAIN_DIR)
        test_path = os.path.join(root_directory, self.TEST_DIR)

        if not (os.path.exists(train_path) or os.path.exists(test_path)):
            print('Parsing raw data to numpy arrays...')
            rawPNG_dir = os.path.join(root_directory, 'raw_png')
            png_to_npy(src_dir=rawPNG_dir, 
                        dest_dir=root_directory, 
                        train_val_split=train_val_split, 
                        seed=seed)        

        split_dir = os.path.join(root_directory, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"'{split}' directory does not exist in {root_directory}")
        self.file_list = [os.path.join(split_dir, f) for f in os.listdir(split_dir) if f.endswith('.npy')]

    def __len__(self):
        return len(self.file_list)

    def _extract_hash_from_path(self, path):
        pattern = r"([a-f0-9]{32})_"  # Regex pattern to match a 32-character hexadecimal string followed by an underscore
        match = re.search(pattern, path)
        if match:
            return match.group(1)
        return "Hash not found"

    def __getitem__(self, idx):
        """
        Returns a video and its associated hash string.
        Shape of video: (T, H, W, C) where T=number of frames, H,W=frame dimensions, C=number of channels
        """
        video_path = self.file_list[idx]
        hash_string = self._extract_hash_from_path(video_path)
        with open(video_path, 'rb') as f:
            video_data = np.load(f)
        return torch.from_numpy(video_data), hash_string



class CollateFnClass:
    """
    A custom collation function for PyTorch DataLoader. The class is responsible for batching,
    segmenting, and padding videos to create uniform-sized inputs for a neural network.

    Attributes:
        chunk_size (int): Size of the chunks into which a video is split.
        seed (int): Seed for random number generation.
        PAD (int): Padding value (currently set to 0).
        split (str): Data split type, e.g., 'train', 'test'.
        segment_flag (bool): Whether to apply segmentation to video frames.
        device (str): The device on which to perform all tensor operations ('cpu' or 'cuda').
    """
    
    def __init__(self, config, split):
        """
        Initializes the CollateFnClass instance.

        Args:
            config (obj): Configuration object containing various settings.
            split (str): The type of data split ('train', 'test', etc.)
        """
        self.chunk_size = config.dataloader.chunk_params.chunk_size
        self.seed = config.dataloader.seed
        self.PAD = 0
        self.split = split
        self.device = 'cpu' 

    def segment_video(self, video):
        segmented_frames = [Segmenter.segment(frame, device=self.device) for frame in video]
        return torch.stack(segmented_frames)
   
    def __call__(self, batch):
        """
        Callable method to perform collation.

        Args:
            batch (list): List of tuples, each containing a video and an associated hash string.

        Returns:
            tuple: Containing the collated batch, padding masks, and hash chunk indices. Batch size= Number of chunks into which the videos are split.
            collated_batch (torch.Tensor): A tensor of shape (B, T, C, H, W) where B=batch size, T=number of chunks, C=number of channels, H=height, and W=width of each frame.
            padding_masks (torch.Tensor): A tensor of shape (B, T) where where B=batch size, T=number of chunks. Each element in the tensor is either 0 or 1, indicating whether frame in chunk is padded or not.
            hash_chunk_indices (list): A list of tuples, each containing chunk information: hash string, chunk index, and total number of chunks in the parent video.
        """
        all_chunks = []
        all_segmented_chunks = [] 
        all_padding_masks = []
        all_hash_chunk_indices = []  # Stores (hash_string, chunk_index, total_chunks) tuples

        for video, hash_string in batch:
            video = video.to(self.device) 
            
            # Segmenting and normalizing the video
            segmented_video = self.segment_video(video)
            video = video / MAX_PIXEL_VALUE
            segmented_video = segmented_video / MAX_PIXEL_VALUE
            assert video.max() <= 1.0 and video.min() >= 0.0, "Video is not normalized!" 
            assert segmented_video.max() <= 1.0 and segmented_video.min() >= 0.0, "Video is not normalized!" 

            video_length = video.shape[0]
            chunks = [video[i:i+self.chunk_size] for i in range(0, video_length, self.chunk_size)]
            segmented_chunks = [segmented_video[i:i+self.chunk_size] for i in range(0, video_length, self.chunk_size)]  
            total_chunks = len(chunks)

            for idx, (chunk, segmented_chunk) in enumerate(zip(chunks, segmented_chunks)):  
                if chunk.shape[0] < self.chunk_size:
                    pad_length = self.chunk_size - chunk.shape[0]
                    pad_tensor = torch.zeros((pad_length, *chunk.shape[1:]), dtype=chunk.dtype, device=self.device) # Padding tensor for chunk
                    segmented_pad_tensor = torch.zeros((pad_length, *segmented_chunk.shape[1:]), dtype=segmented_chunk.dtype, device=self.device)  # Padding tensor for segmented_chunk
                    padding_mask = torch.cat([torch.ones(chunk.shape[0], device=self.device), torch.zeros(pad_length, device=self.device)])

                    chunk = torch.cat([chunk, pad_tensor], dim=0)
                    segmented_chunk = torch.cat([segmented_chunk, segmented_pad_tensor], dim=0)  
                else:
                    padding_mask = torch.ones(self.chunk_size, device=self.device)

                all_chunks.append(chunk)
                all_segmented_chunks.append(segmented_chunk)  
                all_padding_masks.append(padding_mask)
                all_hash_chunk_indices.append((hash_string, idx, total_chunks))

        if self.split == ForestFireDataset.TRAIN_DIR:
            random.seed(self.seed)
            combined_list = list(zip(all_chunks, all_segmented_chunks, all_padding_masks, all_hash_chunk_indices)) 
            random.shuffle(combined_list)
            all_chunks, all_segmented_chunks, all_padding_masks, all_hash_chunk_indices = zip(*combined_list)

        collated_batch = torch.stack(all_chunks, dim=0).permute(0, 1, 4, 2, 3)
        collated_segmented_batch = torch.stack(all_segmented_chunks, dim=0).permute(0, 1, 4, 2, 3)  
        padding_masks = torch.stack(all_padding_masks, dim=0)
        
        
        return collated_batch, padding_masks, all_hash_chunk_indices, collated_segmented_batch  


def get_dataloader(config):
    """
    Initializes and returns training and testing DataLoader instances for the ForestFireDataset.

    This function reads configuration settings from the provided `config` object to set up the DataLoader instances
    for both training and testing purposes. It seeds the random number generator for reproducibility, determines
    the appropriate dataset directory, and configures DataLoader settings like batch size, number of workers, and
    collation function.

    Args:
        config (obj): Configuration object containing settings for data loading, among other things.

    Returns:
        tuple: A tuple containing DataLoader instances for training and testing data.
            - trainloader (torch.utils.data.DataLoader): DataLoader instance for training data.
            - testloader (torch.utils.data.DataLoader): DataLoader instance for testing data.

    Internal Functions:
        - _get_root_directory(dataset_name): Retrieves the root directory path for the given dataset name.
        - _get_dataset(dataset_name, split): Initializes and returns a ForestFireDataset instance.
        - _get_dataloader(dataset, shuffle, batch_size, split, seed): Initializes and returns a DataLoader instance.
        - worker_init_fn(worker_id): Function to seed the worker threads (for reproducibility).
    """
    
    # Seed for reproducibility
    seed = config.dataloader.seed
    torch.manual_seed(seed)
    random.seed(seed)
    
    def _get_root_directory(dataset_name):
        """Retrieve the root directory for the given dataset name."""
        root_directory_mapping = {
            "density76_uniform": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density/uniform_76",
            "density74_uniform": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density/uniform_74",
            "density72_uniform": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density/uniform_72",
            "density70_uniform": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density/uniform_70",
            "density68_uniform": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density/uniform_68",
        
            "density72_p_100_uniform": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_stochastic/uniform_72_p_0",
            "density72_p_95_uniform": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_stochastic/uniform_72_p_95",
            "density72_p_90_uniform": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_stochastic/uniform_72_p_9",
            "density72_p_85_uniform": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_stochastic/uniform_72_p_85",
            "density72_p_80_uniform": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_stochastic/uniform_72_p_8",
            
            "density72_p_100_sameInit_1": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_same_init/uniform_72_p_0",
            "density72_p_95_sameInit_1": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_same_init/uniform_72_p_95",
            "density72_p_90_sameInit_1": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_same_init/uniform_72_p_9",
            "density72_p_85_sameInit_1": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_same_init/uniform_72_p_85",
            "density72_p_80_sameInit_1": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_same_init/uniform_72_p_8",
            
            "density64_p_100_diffDen": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_deterministic/uniform_64_p_0",
            "density66_p_100_diffDen": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_deterministic/uniform_66_p_0",
            "density68_p_100_diffDen": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_deterministic/uniform_68_p_0",
            "density70_p_100_diffDen": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_deterministic/uniform_70_p_0",
            "density72_p_100_diffDen": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_deterministic/uniform_72_p_0",
            "density74_p_100_diffDen": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_deterministic/uniform_74_p_0",
            "density76_p_100_diffDen": "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_deterministic/uniform_76_p_0",
            
        }
        if dataset_name not in root_directory_mapping:
            raise NotImplementedError(f'Unknown dataset: {dataset_name}')
        
        # Generate probability map if sameInit
        if "sameInit" in dataset_name:
            root_directory = root_directory_mapping[dataset_name]
            probability_map_path = os.path.join("./dump/probabilityMap_MC", dataset_name, dataset_name + ".npy")
            os.makedirs(os.path.dirname(probability_map_path), exist_ok=True)
            if not os.path.exists(probability_map_path):
                print(f"Generating probability map for {dataset_name}")
                pmap_config = config
                pmap_config.dataloader.train_val_split = [0,1]
                split = ForestFireDataset.TEST_DIR
                tracker = BurnProbabilityTracker(dataset=ForestFireDataset(args=pmap_config, 
                                                                        root_directory=root_directory, 
                                                                        split=split))
                tracker.compute_burn_probabilities(save_path=probability_map_path)
                
        return root_directory_mapping[dataset_name]

    def _get_dataset(dataset_name, split):
        """Initialize and return a ForestFireDataset instance."""
        root_directory = _get_root_directory(dataset_name)
        return ForestFireDataset(args=config, root_directory=root_directory, split=split)

    def _get_dataloader(dataset, shuffle, batch_size, split, seed):
        """Initialize and return a DataLoader instance."""

        def worker_init_fn(worker_id):
            random_seed = seed + worker_id
            random.seed(random_seed)
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        collate_fn_instance = CollateFnClass(config, split=split)
        return torch.utils.data.DataLoader(
            dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            pin_memory=False,
            num_workers=config.dataloader.num_workers,
            collate_fn=collate_fn_instance,
            worker_init_fn=worker_init_fn if shuffle else None,
        )
    
    trainset = _get_dataset(config.experiment_setting.train.dataset_name, ForestFireDataset.TRAIN_DIR)
    testset = _get_dataset(config.experiment_setting.evaluation.dataset_name, ForestFireDataset.TEST_DIR)

    trainloader = _get_dataloader(trainset, shuffle=True, batch_size=config.dataloader.train_batch_size, split=ForestFireDataset.TRAIN_DIR, seed=seed)
    testloader = _get_dataloader(testset, shuffle=False, batch_size=config.dataloader.test_batch_size, split=ForestFireDataset.TEST_DIR, seed=seed)

    return trainloader, testloader

class TestSuite:
    def __init__(self, config):
        self.config = config
        self.trainloader, self.testloader = get_dataloader(config)
    
    def test_dataset(self):
        print("Testing Dataset:")
        for i in range(3):  # Test 3 samples from the dataset
            video_data, hash_string = self.trainloader.dataset[i]
            print(f"Sample {i+1} shape: {video_data.shape}, Hash: {hash_string}")

    def test_collate_fn(self):
        print("\nTesting Collate Function:")
        collate_fn_instance = CollateFnClass(self.config, split=ForestFireDataset.TRAIN_DIR)
        sample_batch = [self.trainloader.dataset[i] for i in range(3)]  # Take 3 samples to form a mini-batch
        collated_batch, padding_masks, hash_chunk_indices, collated_segmented_batch = collate_fn_instance(sample_batch)  
        print(f"Collated batch shape: {collated_batch.shape}, Padding mask shape: {padding_masks.shape}")
        print(f"Collated segmented batch shape: {collated_segmented_batch.shape}")  
        print(f"Hash Chunk Indices: {hash_chunk_indices} | Total chunks: {len(hash_chunk_indices)}")

    def test_dataloader(self):
        print("\nTesting DataLoader:")
        for i, (data, padding_mask, hash_chunk_indices, segmented_data) in enumerate(self.trainloader):  
            print(f"\n - Batch {i + 1} shape: {data.shape}, Padding mask shape: {padding_mask.shape}")
            print(f"Segmented data shape: {segmented_data.shape}") 
            print(f"Hash Chunk Indices: {hash_chunk_indices} | Total chunks: {len(hash_chunk_indices)}")
            if i == 2:  # Limit to first 3 batches for testing
                break
    
    def visualize_chunks(self, channel_idx):
        print("\nVisualizing Chunks:")

        for i, (data, padding_mask, hash_chunk_indices, segmented_data) in enumerate(self.trainloader):  # Included segmented_data
            print(f"Visualizing Batch {i + 1}")

            for j, (chunk, segmented_chunk, chunk_hash_idx) in enumerate(zip(data, segmented_data, hash_chunk_indices)):  # Included segmented_chunk
                hash_string, chunk_idx, total_chunks = chunk_hash_idx  # Unpack the tuple to get total_chunks

                # Note that chunk shape is (T, C, H, W)
                # Slicing the chunk to keep only the channel at index `channel_idx` for segmented data
                single_channel_chunk = segmented_chunk[:, channel_idx, :, :].cpu().numpy()

                save_path = f'./dump/chunk_visualization/Batch_{i + 1}_Chunk_{j + 1}_Hash_{hash_string}_Idx_{chunk_idx}_Total_{total_chunks}_Channel_{channel_idx}.gif'
                save_gif_from_npy(single_channel_chunk, save_path)

                # Slicing the chunk to keep all channels (assuming BGR or similar) for original data
                bgr_chunk = chunk[:, :, :, :].cpu().numpy()

                save_path = f'./dump/chunk_visualization/Batch_{i + 1}_Chunk_{j + 1}_Hash_{hash_string}_Idx_{chunk_idx}_Total_{total_chunks}_BGR.gif'
                save_gif_from_npy(bgr_chunk, save_path, format='NCHW')

            if i == 0:  # Limit to first batch for visualization
                break


class BurnProbabilityTracker:
    def __init__(self, dataset, device="cpu"):
        self.dataset = dataset
        self.device = device
        self.tracker_video = self._initialize_tracker()

    def _initialize_tracker(self):
        # Find the length of the longest video
        max_length = max([video.shape[0] for video, _ in self.dataset])
        # Initialize the tracker tensor with zeros
        _, H, W, _ = self.dataset[0][0].shape
        tracker = torch.zeros((max_length, H, W), device=self.device)
        return tracker

    def update_tracker(self, video):
        # For each frame in the video, update the tracker_video tensor
        for t, frame in enumerate(video):
            self.tracker_video[t] += frame
            
    def segment_video(self, rgb_video):
        segmented_frames = [Segmenter.segment(frame, device=self.device) for frame in rgb_video]
        return torch.stack(segmented_frames)
    
    def compute_burn_probabilities(self, save_path=None):
        # Process each video in the dataset
        for idx, (video, _) in enumerate(self.dataset):
            print(f"Processing video {idx+1}")
            segmented_video = self.segment_video(video)
            burnt_areas = segmented_video[:, :, :, 0] + segmented_video[:, :, :, 3] 
            burnt_areas = burnt_areas / MAX_PIXEL_VALUE
            save_gif_from_npy(video.cpu().numpy(), "./dump/frame_visualizations/original_video_p100.gif", save_pdf_frames=True)
            # save_gif_from_npy(burnt_areas.cpu().numpy(), "./dump/frame_visualizations/segmented_video_p85.gif", save_pdf_frames=True)
            exit()
            self.update_tracker(burnt_areas)
            
        # Normalize the tracker tensor
        num_videos = len(self.dataset)
        self.tracker_video /= num_videos

        # Save the tracker video as a GIF and a numpy array
        if save_path is not None:
            save_gif_from_npy(self.tracker_video.cpu().numpy(), save_path.replace(".npy", ".gif"))
            np.save(save_path, self.tracker_video.cpu().numpy())
            
        return self.tracker_video

def main():
    # Configuration Loading
    base_config = load_config('./config/base_config.yaml')
    update_config = load_config('./config/update_config.yaml')
    config = merge_configs(base_config, update_config)
    
    print("Final Configuration:")
    print(yaml.dump(dict_recursive(config)))
    
    # # Initialize test suite
    # test_suite = TestSuite(config)
    
    # # Run individual tests
    # test_suite.test_dataset()
    # test_suite.test_collate_fn()
    # test_suite.test_dataloader()
    
    # # Visualize chunks 
    # channel_index_mapping = {"active_fire": 0, "vegetation": 1, "empty": 2, "ember": 3}
    # test_suite.visualize_chunks(channel_idx=channel_index_mapping["ember"])
    
    # Burn Probability Tracker
    config.dataloader.train_val_split = [0,1]
    root_directory = "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density_same_init/uniform_72_p_0"
    split = ForestFireDataset.TEST_DIR
    tracker = BurnProbabilityTracker(dataset=ForestFireDataset(args=config, 
                                                               root_directory=root_directory, 
                                                               split=split))
    burn_probabilities = tracker.compute_burn_probabilities()
    print(f"Shape of burn_probabilities: {burn_probabilities.shape}")
    save_gif_from_npy(burn_probabilities.cpu().numpy(), "./dump/frame_visualizations/burn_probabilities_p_85.gif", save_pdf_frames=True)


if __name__ == "__main__":
    main()


        
   