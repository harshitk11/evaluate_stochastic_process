# utils/checkpoint_utils.py

import torch
import os

def save_checkpoint(state, filename):
    """
    Save model and optimizer states as a checkpoint.
    """
    try:
        torch.save(state, filename)
        print(f"Checkpoint saved successfully to {filename}")
    except Exception as e:
        print(f"Failed to save checkpoint to {filename}: {e}")

def load_checkpoint(filename, model, optimizer=None):
    """
    Load model and optionally optimizer states from a checkpoint.
    """
    try:
        checkpoint = torch.load(filename)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        epoch = checkpoint['epoch']
        loss = checkpoint.get('loss', None)  # it's okay if loss isn't available

        print(f"Checkpoint loaded successfully from {filename} at epoch {epoch}")
        return model, optimizer, epoch, loss
    
    except Exception as e:
        print(f"Failed to load checkpoint from {filename}: {e}")
        return model, optimizer, 0, None

def initialize_or_load_checkpoint(model, optimizer, checkpoint_path=None, load_only_model=False):
    """
    Initialize or load a model from a checkpoint.
    Useful during the start of training.
    """
    start_epoch = 0
    if checkpoint_path and os.path.exists(checkpoint_path):
        if load_only_model:
            model, _, _, _ = load_checkpoint(checkpoint_path, model)
        else:
            model, optimizer, start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer)

    else:
        print(f"Checkpoint path does not exist: {checkpoint_path}")
        # raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")
    
    return model, optimizer, start_epoch

def main():
    pass

if __name__ == "__main__":
    main()
