import numpy as np

def create_mask(img_t, img_t1, exclude_value=-1):
    mask_t = img_t != exclude_value
    mask_t1 = img_t1 != exclude_value
    combined_mask = np.logical_and(mask_t, mask_t1)
    return combined_mask

def jaccard_similarity(img_t_masked, img_t1_masked):
    intersection = np.logical_and(img_t_masked, img_t1_masked).sum()
    union = np.logical_or(img_t_masked, img_t1_masked).sum()
    
    return intersection / union if union != 0 else 0

def dice_similarity(img_t_masked, img_t1_masked):
    intersection = np.logical_and(img_t_masked, img_t1_masked).sum()
    total = img_t_masked.sum() + img_t1_masked.sum()
    
    return (2. * intersection) / total if total != 0 else 0

def calculate_similarity_metrics(inputs, labels, batch_index=0):
    metrics = {}
    
    img_t = inputs[batch_index, :, :, -1]  # Assuming "PrevFireMask" is the last channel
    img_t1 = labels[batch_index, :, :, 0]
    assert img_t.shape == img_t1.shape, f'Input and label shapes do not match ({img_t.shape} vs {img_t1.shape})'
    assert set(np.unique(img_t)).issubset([0, 1, -1]), f'Input image contains values other than 0, 1, and -1'
    assert set(np.unique(img_t1)).issubset([0, 1, -1]), f'Input image contains values other than 0, 1, and -1'
    
    mask = create_mask(img_t, img_t1)
    img_t_masked = img_t[mask]
    img_t1_masked = img_t1[mask]
    assert set(np.unique(img_t_masked)).issubset([0, 1]), f"Input image contains values other than 0, 1"
    assert set(np.unique(img_t1_masked)).issubset([0, 1]), f'Input image contains values other than 0, 1'
        
    # Calculate and store each metric
    metrics['Jaccard_Similarity'] = jaccard_similarity(img_t_masked, img_t1_masked)
    metrics['Dice_Similarity'] = dice_similarity(img_t_masked, img_t1_masked)
    
    return metrics


def main():
    pass

if __name__ == '__main__':
    main()