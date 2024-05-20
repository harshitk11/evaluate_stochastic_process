"""
Visualize the distribution of the model predictions
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.stats import entropy

# Function to plot RGB images
def plot_rgb_images(images, ax, title, skip_frames=5):
    for idx, img in enumerate(images):
        img_rgb = cv2.cvtColor(img.permute(1, 2, 0).cpu().detach().numpy(), cv2.COLOR_BGR2RGB)
        ax[idx].imshow(img_rgb)
        ax[idx].axis('off')
        ax[idx].set_title(f'Frame {idx*skip_frames + 1}', fontsize=10)
    ax[0].set_ylabel(title, fontsize=16)

# Function to plot probability maps
def plot_probability_maps(prob_maps, ax, title, skip_frames=5):
    for idx, prob_map in enumerate(prob_maps):
        ax[idx].imshow(prob_map.squeeze().cpu().detach().numpy(), cmap='viridis')
        ax[idx].axis('off')
        ax[idx].set_title(f'Frame {idx*skip_frames + 1}', fontsize=10)
    ax[0].set_ylabel(title, fontsize=16)
        
def plot_histograms(prob_maps, save_path_prefix, skip_frames=5):
    for idx, prob_map in enumerate(prob_maps):
        flat_probs = prob_map.cpu().detach().numpy().flatten()
        
        # Applying the transformation |x - 0.5| to each value in flat_probs
        transformed_probs = np.abs(flat_probs - 0.5)
        
        # Calculating the standard deviation of the transformed probabilities
        std_dev = np.std(transformed_probs)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(transformed_probs, bins=20, density=True, color='blue', edgecolor='black', alpha=0.7)
        ax.set_xlim([0, 0.5])  # Adjusted xlim due to the transformation
        # ax.set_ylim([0, 5])
    
        ax.set_title(f'Transformed Probability Distribution for Frame {idx*skip_frames + 1}', fontsize=16)
        ax.set_xlabel('Transformed Probability', fontsize=14)
        ax.set_ylabel('Density', fontsize=14)
        
        # Add a grid
        ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
        
        # Annotate standard deviation
        ax.annotate(f'Standard Deviation: {std_dev:.2f}', xy=(0.05, 4), fontsize=12)  # Adjust xy coordinates as needed
        
        # Set tick label size
        ax.tick_params(axis='both', which='major', labelsize=12)

        # Save the histogram as a high-quality PNG file
        save_path = f"{save_path_prefix}_Frame_{idx*skip_frames + 1}.png"
        plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
        
        plt.close()


# Function to visualize frames, probability maps, and histograms
def visualize_distribution_predictions(bgr_gt, prediction, save_path, skip_frames=5):
    # Select the frames to visualize (every skip_frames frame along the T axis)
    selected_frames = [bgr_gt[0, i] for i in range(0, bgr_gt.shape[1], skip_frames)]
    selected_prob_maps = [prediction[0, i] for i in range(0, prediction.shape[1], skip_frames)]

    # Generate histograms and save them as separate PNGs
    plot_histograms(selected_prob_maps, save_path_prefix=save_path, skip_frames=skip_frames)
