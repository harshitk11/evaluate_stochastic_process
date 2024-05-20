from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors



def save_gif_from_npy(video_npy, save_path, format='NHWC', save_pdf_frames=False):    
    """
    Save a GIF from a numpy array of video frames.
    params:
        video_npy (np.ndarray): Numpy array of video frames, of shape (Frames, Height, Width, Channels)
        save_path (str): Path to save the GIF
        format (str): Format of the numpy array, either 'NHWC' or 'NCHW'
    """
    
    # Determine if the video is single-channel (grayscale) or multi-channel (color)
    is_single_channel = len(video_npy.shape) == 3  # Shape should be (Frames, Height, Width) for grayscale
    
    if format == 'NCHW' and not is_single_channel:
        video_npy = video_npy.transpose(0, 2, 3, 1)  # Change to NHWC
    
    if save_pdf_frames:
        for idx, frame in enumerate(video_npy):
            # Save a frame as PDF (assumes BGR)
            frame_path = save_path.replace(".gif", f"_{idx}.pdf")
            
            if is_single_channel:
                # If grayscale, convert frame to 2D array
                frame = np.squeeze(frame)
                frame = frame[-10:10, -10:10]
                plt.imshow(frame, cmap='gray')
            else:
                # If BGR, convert to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # start_indx = 20
                # frame = frame[start_indx:start_indx+8, start_indx:start_indx+8, :]
                # print(cropped_frame.shape, frame.shape)
                plt.imshow(frame)
            
            plt.axis('off')
            plt.savefig(frame_path, bbox_inches='tight', pad_inches=0, format='pdf')
            plt.close()
        
    # Convert numpy frames to PIL Images
    if is_single_channel:
        pil_images = [Image.fromarray((frame * 255.0).astype(np.uint8), 'L') for frame in video_npy]  # 'L' mode for grayscale
    else:
        # Convert from BGR to RGB
        rgb_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) for frame in video_npy]
        pil_images = [Image.fromarray((frame * 255.0).astype(np.uint8), 'RGB') for frame in rgb_frames]
    
    pil_images[0].save(
        save_path,
        append_images=pil_images[1:],
        save_all=True,
        duration=100,  # Duration for each frame, in milliseconds
        loop=1,  # 0 for infinite loop
        dither='NONE'
    )
    
def plot_images_in_row(images, ax, title, is_segmented=False, add_frame_numbers=False, skip_frames=5, row_label=None, threshold_value=None):
    for idx, img in enumerate(images):
        if not is_segmented:  # Convert from BGR to RGB for non-segmented images
            img = cv2.cvtColor(img.permute(1, 2, 0).cpu().detach().numpy(), cv2.COLOR_BGR2RGB)
        else:  # For segmented images, no need to convert channels
            img = img.squeeze().cpu().detach().numpy()

        ax[idx].imshow(img, cmap='gray' if is_segmented else None)
        ax[idx].axis('off')
        if add_frame_numbers:
            ax[idx].set_title(f'Frame {idx*skip_frames + 1}', fontsize=10)
        if threshold_value is not None:
            ax[idx].set_title(f"{threshold_value[idx]:.2f}", fontsize=10)
            # print(f"{threshold_value[idx]:.2f}")
            
            
    if row_label is not None:
        ax[0].text(-2*img.shape[1]//3, img.shape[0]//2, row_label, fontsize=12, ha='center', va='center')
        
    ax[0].set_ylabel(title, fontsize=16)

    
    
def visualize_frames(bgr_gt, segmented_gt, prediction, channel_idx, save_path, skip_frames=5):
    # Select the frames to visualize (every skip_frames frame along the T axis)
    selected_frames = [bgr_gt[0, i] for i in range(0, bgr_gt.shape[1], skip_frames)]
    selected_segmented_frames = [segmented_gt[0, i, channel_idx:channel_idx+1] for i in range(0, segmented_gt.shape[1], skip_frames)]
    selected_output_frames = [prediction[0, i] for i in range(0, prediction.shape[1], skip_frames)]
    
    # Threshold the selected_output_frames
    threshold_value = 0.5
    selected_output_frames_thresholded = [(frame > threshold_value).float() for frame in selected_output_frames]

    # Determine the number of columns needed for the plot
    num_cols = len(selected_frames)
    
    # Create a subplot with 4 rows and num_cols columns
    fig, axes = plt.subplots(4, num_cols, figsize=(15, 5))

    # Plot the frames in the respective rows
    plot_images_in_row(selected_frames, axes[0], 'Collated Batch (Ground Truth)', is_segmented=False, add_frame_numbers=True, skip_frames=skip_frames)
    plot_images_in_row(selected_segmented_frames, axes[1], 'Segmented Collated Batch', is_segmented=True, skip_frames=skip_frames)
    plot_images_in_row(selected_output_frames, axes[2], 'Output (Lowest Scale)', is_segmented=True, skip_frames=skip_frames)
    plot_images_in_row(selected_output_frames_thresholded, axes[3], 'Output Thresholded (0.5)', is_segmented=True, skip_frames=skip_frames)

    # Save the plot as a PNG file
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(save_path)
    plt.close()

def plot_grayscale_image(np_array, save_path):
    plt.figure()
    plt.imshow(np_array, cmap='gray')
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, format='pdf')
    plt.close()

def visualize_frames_different_thresholds(bgr_gt, segmented_gt, prediction, channel_idx, save_path, skip_frames=5, threshold_values=[0.1, 0.3, 0.5, 0.7, 0.9], gt_probability_map=None, optimal_threshold=None):    
    # Select the frames to visualize (every skip_frames frame along the T axis)
    selected_frames = [bgr_gt[0, i] for i in range(0, bgr_gt.shape[1], skip_frames)]
    selected_segmented_frames = [segmented_gt[0, i, channel_idx:channel_idx+1] for i in range(0, segmented_gt.shape[1], skip_frames)]
    selected_output_frames = [prediction[0, i] for i in range(0, prediction.shape[1], skip_frames)]
    
    idx = -2
    frame_path_gt = save_path.replace(".png", f"_segmented_gt_{idx}.pdf")
    frame_path_pred = save_path.replace(".png", f"_pred_{idx}.pdf")
    frame_path_gt_dist = save_path.replace(".png", f"_gt_dist_{idx}.pdf")
    plot_grayscale_image(selected_segmented_frames[idx].detach().cpu().squeeze(), frame_path_gt)
    plot_grayscale_image(selected_output_frames[idx].detach().cpu().squeeze(), frame_path_pred)
    
    selected_output_frames_thresholded = {}
    for threshold_value in threshold_values:
        # print(f"Thresholding at {threshold_value}")
        selected_output_frames_thresholded[threshold_value] = [(frame > threshold_value).float() for frame in selected_output_frames]
    
    # Optimal threshold based visualization
    if optimal_threshold is not None:
        selected_optimal_threshold_roc = [optimal_threshold["roc"][i] for i in range(0, len(optimal_threshold["roc"]), skip_frames)]
        selected_optimal_threshold_pr = [optimal_threshold["pr"][i] for i in range(0, len(optimal_threshold["pr"]), skip_frames)]
        selected_output_frames_thresholded["roc"] = [(frame > selected_optimal_threshold_roc[indx]).float() for indx, frame in enumerate(selected_output_frames)]
        selected_output_frames_thresholded["pr"] = [(frame > selected_optimal_threshold_pr[indx]).float() for indx, frame in enumerate(selected_output_frames)]
        
    # Determine the number of columns needed for the plot
    num_cols = len(selected_frames)
    
    # Create a subplot with rows and num_cols columns
    threshold_values_total = threshold_values + ["roc", "pr"]
    fig, axes = plt.subplots(3+ len(threshold_values_total), num_cols, figsize=(25, 25))

    start_indx = 0
    if gt_probability_map is not None:
        start_indx = 1
        fig, axes = plt.subplots(4 + len(threshold_values_total), num_cols, figsize=(25, 25))
        selected_gt_probability_map = [gt_probability_map[i] for i in range(0, bgr_gt.shape[1], skip_frames)]
        plot_grayscale_image(selected_gt_probability_map[idx].detach().cpu().squeeze(), frame_path_gt_dist)
        plot_images_in_row(selected_gt_probability_map, axes[0], 'Prob Map (Ground Truth)', is_segmented=True, skip_frames=skip_frames, row_label='Prob Map GT')

    # Plot the frames in the respective rows
    plot_images_in_row(selected_frames, axes[start_indx], 'Collated Batch (Ground Truth)', is_segmented=False, add_frame_numbers=True, skip_frames=skip_frames, row_label='RGB')
    start_indx += 1
    plot_images_in_row(selected_segmented_frames, axes[start_indx], 'Segmented Collated Batch', is_segmented=True, skip_frames=skip_frames, row_label='Segmented GT')
    start_indx += 1
    plot_images_in_row(selected_output_frames, axes[start_indx], 'Output (Lowest Scale)', is_segmented=True, skip_frames=skip_frames, row_label='Output (p)')
    start_indx += 1
    for threshold_idx, threshold_value in enumerate(threshold_values_total):
        row_label = f'Thr {threshold_value}'
        if threshold_value == "roc":
            plot_images_in_row(selected_output_frames_thresholded[threshold_value], axes[start_indx + threshold_idx], row_label, is_segmented=True, skip_frames=skip_frames, row_label=row_label, threshold_value=selected_optimal_threshold_roc)
        elif threshold_value == "pr":
            plot_images_in_row(selected_output_frames_thresholded[threshold_value], axes[start_indx + threshold_idx], row_label, is_segmented=True, skip_frames=skip_frames, row_label=row_label, threshold_value=selected_optimal_threshold_pr)
        else:
            plot_images_in_row(selected_output_frames_thresholded[threshold_value], axes[start_indx + threshold_idx], row_label, is_segmented=True, skip_frames=skip_frames, row_label=row_label)
            
    # Save the plot as a PNG file
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.1)
    plt.savefig(save_path)
    plt.close()
    
    
def plot_channels_labels_and_output(inputs, labels, output, batch_index=0, save_path=None, only_prev_fire_mask=False, performance_score=None, stochasticity_estimate=None):
    """
    Plots each channel of the inputs as grayscale images, labels, and outputs beside it.
    
    Parameters:
    - inputs: tensor, shape = (batch_size, side_length, side_length, num_in_channels)
    - labels: tensor, shape = (batch_size, side_length, side_length, 1)
    - output: tensor, shape = (batch_size, side_length, side_length, 1)
    - batch_index: index of the image in the batch to be visualized
    """
    INPUT_FEATURES = ['elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph', 
                      'pr', 'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask']
    
    # Create a colormap for the labels and PrevFireMask
    cmap = mcolors.ListedColormap(['grey', 'black', 'white'])
    bounds = [-1.5, -0.5, 0.5, 1.5]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Deciding which channels to plot
    channels_to_plot = [-1] if only_prev_fire_mask else range(inputs.shape[-1])
    
    # Setting up the subplot grid
    fig, axs = plt.subplots(1, len(channels_to_plot)+2, figsize=(17, 5))  # adjusted the size and columns
    
    for i, channel_index in enumerate(channels_to_plot):
        # Plotting each chosen channel
        img = inputs[batch_index, :, :, channel_index]
        
        if INPUT_FEATURES[channel_index] in ['PrevFireMask']:
            unique_vals = np.unique(img)
            for val in unique_vals:
                if val not in [-1, 0, 1]:
                    raise ValueError("Invalid value in label or PrevFireMask.")
            
            axs[i].imshow(img, cmap=cmap, norm=norm)
            print(f"\nUnique values in {INPUT_FEATURES[channel_index]}: {unique_vals}")
        else:
            axs[i].imshow(img, cmap='gray')
            
        axs[i].axis('off')
        axs[i].set_title(f'{INPUT_FEATURES[channel_index]}')
    
    # Plotting the label
    axs[-2].imshow(labels[batch_index, :, :, 0], cmap=cmap, norm=norm)
    axs[-2].axis('off')
    if stochasticity_estimate is not None:
        axs[-2].set_title(f'DC: {stochasticity_estimate["Dice_Similarity"]:.2f}')
    else:
        axs[-2].set_title(f'GT')
    # Decrease the space between this plot and the plot on the left
    plt.subplots_adjust(wspace=0.05)
    
    print(f"Unique values in GT: {np.unique(labels[batch_index, :, :, 0])} with fire pixels: {np.sum(labels[batch_index, :, :, 0] == 1)}")
    
    # Plotting the output and show colorbar
    im = axs[-1].imshow(output[batch_index, :, :, 0], cmap='Reds', vmin=0, vmax=1)
    axs[-1].axis('off')
    axs[-1].set_title(f'AUC:{performance_score["auc_pr"]:.3f},MSE:{performance_score["mse"]:.3f},Rec:{performance_score["recall"]:.3f},Pr:{performance_score["precision"]:.3f}')
    
    cbar = fig.colorbar(im, ax=axs[-1])
    cbar.ax.tick_params(labelsize=14)
    # cbar.set_label('Forecasted Probability', rotation=270, labelpad=15)
    
    plt.subplots_adjust(wspace=0.05)
    # # Plot the thresholded output
    # thresholded_output = (output > 0.5)
    # axs[-1].imshow(thresholded_output[batch_index, :, :, 0], cmap=cmap, norm=norm)
    # axs[-1].axis('off')
    # axs[-1].set_title(f'Thresholded Output')
    
    # Displaying the plot
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    pass  