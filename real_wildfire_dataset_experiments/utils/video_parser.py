import os
import re
import random
from PIL import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import torch
MAX_PIXEL_VALUE = 255.0


class Segmenter:
    # the channels are (b, g, r)
    BGR_RED = torch.Tensor([41, 50, 215])
    BGR_BLACK = torch.Tensor([0, 0, 0])
    BGR_GREEN = torch.Tensor([60, 176, 89])

    @classmethod
    def segment(cls, x, device='cuda'):
        """
        Segments the image based on the defined color channels using PyTorch.
        
        Args:
            x (torch.Tensor): Input RGB image on GPU.
            
        Returns:
            torch.Tensor: A 4-channel binary image where each channel corresponds to 
                          active fire, vegetation, empty, and ember (all other segments).
        """
        BGR_RED = cls.BGR_RED.to(device)
        BGR_GREEN = cls.BGR_GREEN.to(device)
        BGR_BLACK = cls.BGR_BLACK.to(device)

        active_fire = (x == BGR_RED).all(dim=-1)
        vegetation = (x == BGR_GREEN).all(dim=-1)
        empty = (x == BGR_BLACK).all(dim=-1)
        ember = ~(active_fire + vegetation + empty)
        
        # Convert boolean tensor to grayscale convention (0 or 255)
        active_fire = (active_fire * MAX_PIXEL_VALUE).float()
        vegetation = (vegetation * MAX_PIXEL_VALUE).float()
        empty = (empty * MAX_PIXEL_VALUE).float()
        ember = (ember * MAX_PIXEL_VALUE).float()
        
        segmented_img = torch.stack((active_fire, vegetation, empty, ember), dim=-1)
        
        return segmented_img
    
    @classmethod
    def test_segmentation(cls, frame, file_prefix='test'):
        if frame.max() <= 1:
            frame = (frame * MAX_PIXEL_VALUE).astype(np.uint8)
        active_fire, vegetation, empty, ember = np.split(frame, 4, axis=-1)
        
        # Plotting
        fig, axs = plt.subplots(3, 2, figsize=(10, 15))  # 3 rows, 2 columns
        axs = axs.ravel()

        axs[0].imshow(frame)
        axs[0].set_title('Original')
        axs[1].imshow(MAX_PIXEL_VALUE*active_fire, cmap='gray')
        axs[1].set_title('Active Fire')
        axs[2].imshow(MAX_PIXEL_VALUE*vegetation, cmap='gray')
        axs[2].set_title('Vegetation')
        axs[3].imshow(MAX_PIXEL_VALUE*empty, cmap='gray')
        axs[3].set_title('Empty')
        axs[4].imshow(MAX_PIXEL_VALUE*ember, cmap='gray')
        axs[4].set_title('Ember')
        axs[5].imshow(MAX_PIXEL_VALUE*(ember + active_fire), cmap='gray')
        axs[5].set_title('Ember + Active Fire')

        for ax in axs:
            ax.axis('off')

        plt.tight_layout()
        plt.savefig(dpi=300, fname=f'./{file_prefix}.png')
        plt.close()


def extract_info_from_filename(filename):
    match = re.match(r"experiment_(\w+?)_(\d+?)_(\d+?).png", filename)
    if match:
        unique_hash, video_index, frame_number = match.groups()
    else:
        raise ValueError(f"Filename {filename} does not match regex.")
    return unique_hash, int(video_index), int(frame_number)

def png_to_npy(src_dir, dest_dir, train_val_split=(0.7, 0.3), seed=None):    
    """
    Convert a directory of PNG files to numpy arrays and split them into training and testing sets.
    
    The filenames are expected to be of the format: experiment_<hash>_<video_index>_<frame_number>.png
    They are grouped by <hash>_<video_index> and saved as individual numpy arrays.
    
    Args:
        src_dir (str): Path to the directory containing PNG files.
        dest_dir (str): Destination directory where numpy arrays will be saved.
        train_val_split (tuple, optional): Ratio of train and test split. Defaults to (0.7, 0.3).
        seed (int, optional): Seed for reproducibility. Defaults to None.
    
    Raises:
        ValueError: If train_val_split doesn't sum up to 1.0.
    """
    assert sum(train_val_split) == 1.0, "Train-val split should sum up to 1.0"

    # Group by unique video
    video_dict = {}
    for filename in os.listdir(src_dir):
        info = extract_info_from_filename(filename)
        if info:
            unique_hash, video_index, frame_number = info
            video_key = f"{unique_hash}_{video_index}"
            if video_key not in video_dict:
                video_dict[video_key] = []
            video_dict[video_key].append((frame_number, filename))
        else:
            raise ValueError(f"Filename {filename} does not match regex.")

    video_npy_list = []  # List to hold numpy arrays and their IDs

    # Process each video group
    for video_key, frames in video_dict.items():
        frames.sort()  # Sort by frame number
        stacked_frames = []

        for frame_num, filename in frames:
            try:
                img = Image.open(os.path.join(src_dir, filename))
                temp_np_img = np.asarray(img)[:,:,:3]
                
                # NOTE: Convert RGB to BGR
                np_img = np.copy(temp_np_img) #RGB
                np_img[:,:,0] = temp_np_img[:,:,2] #BGR
                np_img[:,:,2] = temp_np_img[:,:,0] #BGR
                
                # Segmenter.test_segmentation(np_img, file_prefix=f"{video_key}_{frame_num}") ## For visualizing segmentation
                stacked_frames.append(np_img)
            except Exception as e:
                print(f"Error processing {os.path.join(src_dir, filename)}: {e}")
                continue

        video_npy = np.stack(stacked_frames, axis=0)
        video_npy_list.append({'id': video_key, 'data': video_npy})
        
    # Shuffle and split the video list
    if seed is not None: 
        random.seed(seed)
    random.shuffle(video_npy_list)
    split_index = int(len(video_npy_list) * train_val_split[0])
    train_videos = video_npy_list[:split_index]
    val_videos = video_npy_list[split_index:]

    # Save videos to train and test directories
    print("Saving training videos ...")
    for video in train_videos:
        save_path = os.path.join(dest_dir, 'Train', f"experiment_{video['id']}.npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, video['data'])
        print(f"Saved {video['id']} | Shape: {video['data'].shape}")
        
    print("\nSaving testing videos ...")
    for video in val_videos:
        save_path = os.path.join(dest_dir, 'Test', f"experiment_{video['id']}.npy")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.save(save_path, video['data'])
        print(f"Saved {video['id']} | Shape: {video['data'].shape}")



if __name__ == "__main__":
    src_dir = "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density/uniform_76/raw_png"
    dest_dir = "/ssd_2tb/hkumar64/datasets/netlogo_simulations/uniform_density/uniform_76"
    png_to_npy(src_dir, dest_dir, train_val_split=(0.7, 0.3), segment=True)
      