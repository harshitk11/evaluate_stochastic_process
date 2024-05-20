import re
from typing import Dict, List, Optional, Text, Tuple
import matplotlib.pyplot as plt
from matplotlib import colors

import tensorflow as tf
from wildfire_utils.parser import get_dataset

import matplotlib.pyplot as plt
import numpy as np
from wildfire_utils.constants import DATA_STATS, INPUT_FEATURES, OUTPUT_FEATURES

def plot_channels_and_labels(inputs, labels, batch_index=0):
    """
    Plots each channel of the inputs as grayscale images and the labels beside it.
    
    Parameters:
    - inputs: tensor, shape = (batch_size, side_length, side_length, num_in_channels)
    - labels: tensor, shape = (batch_size, side_length, side_length, 1)
    - batch_index: index of the image in the batch to be visualized
    """
    
    num_in_channels = inputs.shape[-1]
    
    # Setting up the subplot grid
    fig, axs = plt.subplots(1, num_in_channels+1, figsize=(15, 5))
    
    for i in range(num_in_channels):
        # Plotting each channel
        axs[i].imshow(inputs[batch_index, :, :, i], cmap='gray')
        axs[i].axis('off')
        axs[i].set_title(f'{INPUT_FEATURES[i]}')
    
    # Plotting the label beside the last channel
    axs[-1].imshow(labels[batch_index, :, :, 0], cmap='gray')
    axs[-1].axis('off')
    axs[-1].set_title('Label')
    
    # Displaying the plot
    plt.tight_layout()
    plt.savefig('sample_input_and_label.png')

def get_dataloader(config):
    side_length = 64 #length of the side of the square you select (so, e.g. pick 64 if you don't want any random cropping)
    num_obs = 100 #batch size
    file_pattern_train = '/ssd_2tb/hkumar64/datasets/next_day_wildfire_spread/next_day_wildfire_spread_train*'
    file_pattern_eval = '/ssd_2tb/hkumar64/datasets/next_day_wildfire_spread/next_day_wildfire_spread_eval*'
    file_pattern_test = '/ssd_2tb/hkumar64/datasets/next_day_wildfire_spread/next_day_wildfire_spread_test*'

    dataset_train = get_dataset(file_pattern_train,
        data_size=64,
        sample_size=side_length,
        batch_size=num_obs,
        num_in_channels=12,
        compression_type=None,
        clip_and_normalize=True,
        clip_and_rescale=False,
        random_crop=True,
        center_crop=False)
    dataset_eval = get_dataset(file_pattern_eval,
        data_size=64,
        sample_size=side_length,
        batch_size=num_obs,
        num_in_channels=12,
        compression_type=None,
        clip_and_normalize=True,
        clip_and_rescale=False,
        random_crop=False,
        center_crop=False)
    dataset_test = get_dataset(file_pattern_test,
        data_size=64,
        sample_size=side_length,
        batch_size=1,
        num_in_channels=12,
        compression_type=None,
        clip_and_normalize=True,
        clip_and_rescale=False,
        random_crop=False,
        center_crop=False)
    
    return dataset_train, dataset_eval, dataset_test

def main():
    side_length = 64 #length of the side of the square you select (so, e.g. pick 64 if you don't want any random cropping)
    num_obs = 100 #batch size
    file_pattern = '/ssd_2tb/hkumar64/datasets/next_day_wildfire_spread/next_day_wildfire_spread_train*'

    dataset = get_dataset(
        file_pattern,
        data_size=64,
        sample_size=side_length,
        batch_size=num_obs,
        num_in_channels=12,
        compression_type=None,
        clip_and_normalize=False,
        clip_and_rescale=True,
        random_crop=True,
        center_crop=False)
    
    inputs, labels = next(iter(dataset))
    print(inputs.shape) # (batch_size, side_length, side_length, num_in_channels)
    print(labels.shape) # (batch_size, side_length, side_length, 1)
    # Print min and max value in inputs and lables
    print(f'Min value in inputs: {np.min(inputs)}')
    print(f'Max value in inputs: {np.max(inputs)}')
    print(f'Min value in labels: {np.min(labels)}')
    print(f'Max value in labels: {np.max(labels)}')
    
    plot_channels_and_labels(inputs, labels, batch_index=2)


if __name__ == '__main__':
    main()