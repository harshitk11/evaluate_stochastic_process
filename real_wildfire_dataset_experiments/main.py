# main.py

import torch
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os
import yaml
import torch.optim as optim
import argparse

# Local imports
import sys
sys.path.append('./model')
from dataset import get_dataloader
from model.model import BaseModel_WithMultiScale
from model.base_models.segmentation_model import SegmentationModel
from utils.config_utils import load_config, merge_configs, save_config, dict_recursive
from utils.checkpoint_utils import save_checkpoint, initialize_or_load_checkpoint
from train_and_eval import Trainer

def train_and_eval_model(config, epoch, model, optimizer, scheduler, trainloader, testloader, writer=None, log_file=None):
    trainer = Trainer(config, model, optimizer, trainloader, testloader, writer)
    train_metrics = trainer.train_step(epoch)
    eval_metrics = trainer.eval_step(epoch)
    scheduler.step()
    
    if writer:
        writer.add_scalar('Loss/train', train_metrics['train_loss'], epoch)
        writer.add_scalar('Loss/eval', eval_metrics['val_loss'], epoch)
        
    if log_file:
        log_file.write(
            f"Epoch {epoch}: Training Loss = {train_metrics['train_loss']}, "
            f"Validation Loss = {eval_metrics['val_loss']}, "
            f"Filtered Samples = {train_metrics['num_filtered_samples']}, "
            f"Total Samples = {train_metrics['total_samples']}\n"
        )
        
        if eval_metrics["evaluation_scores"] is not None:
            for metric, scores in eval_metrics['evaluation_scores'].items():
                log_file.write(f"Epoch {epoch}: {metric} = ")
                for class_label, score in scores.items():
                    log_file.write(f"Class {class_label}: {score:.4f}, ")
                log_file.write("\n")
            
    return train_metrics, eval_metrics

def load_configurations(base_config_path, update_config_path):
    base_config = load_config(base_config_path)
    update_config = load_config(update_config_path)
    config = merge_configs(base_config, update_config)
    print(" --------- Final Configuration ---------")
    print(yaml.dump(dict_recursive(config))) 
    print(" ---------------------------------------")
    return base_config, update_config, config


def initialize_logging(config):
    LOGS_FLAG = config.experiment_setting.logs
    EXPERIMENT_NAME = config.experiment_setting.experiment_name
    EVAL_ONLY = config.experiment_setting.evaluation.eval_only
    
    writer, log_dir, log_file = None, None, None
    if LOGS_FLAG: 
        if EXPERIMENT_NAME is not None:
            experiment_name = EXPERIMENT_NAME
        else:
            experiment_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
       
        if EVAL_ONLY:
            log_dir = os.path.join("artifacts", "evaluations", experiment_name)
        else:
            log_dir = os.path.join("artifacts", experiment_name)
            
        os.makedirs(log_dir, exist_ok=True)
        
        if not EVAL_ONLY:
            writer = SummaryWriter(log_dir)
        
        # Create a log file for storing epoch information
        log_file = open(os.path.join(log_dir, 'epoch_logs.txt'), 'w')
        
    return writer, log_dir, log_file

def initialize_model_and_optimizer(config, DEVICE, writer=None):
    BASE_MODEL_NAME = config.model.name
    
    if BASE_MODEL_NAME == "segmentation_model":
        USE_BOTTLENECK = config.model.segmentation_model.use_bottleneck
        model = SegmentationModel(use_bottleneck=USE_BOTTLENECK)
    elif BASE_MODEL_NAME == "d_convlstm" or BASE_MODEL_NAME == "depth_convlstm_multiscale":
        model = BaseModel_WithMultiScale(config=config, base_model=BASE_MODEL_NAME, writer=writer)
    else:
        raise ValueError("Model name not recognized.")
    model = model.to(DEVICE)
    LR = config.experiment_setting.train.optimizer.lr
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    return model, optimizer

def initialize_scheduler(optimizer, config):
    GAMMA = config.experiment_setting.train.scheduler.gamma
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=GAMMA)
    return scheduler

def main():
    parser = argparse.ArgumentParser(description="Load configurations")
    parser.add_argument('--base_config', type=str, default='./config/base_config.yaml', help='Path to the base configuration file')
    parser.add_argument('--update_config', type=str, default='./config/update_config.yaml', help='Path to the update configuration file')

    args = parser.parse_args()

    # Load Configurations
    base_config, update_config, config = load_configurations(args.base_config, args.update_config)

    
    # Setup Logging
    writer, log_dir, log_file = initialize_logging(config)
    if log_dir:
        # Save the configuration files to the log directory 
        save_config(config=dict_recursive(config), file_path=os.path.join(log_dir, 'config.yaml'))
        save_config(config=dict_recursive(base_config), file_path=os.path.join(log_dir, 'base_config.yaml'))
        save_config(config=dict_recursive(update_config), file_path=os.path.join(log_dir, 'update_config.yaml'))
    
    # Config Variables
    DEVICE = config.dataloader.device
    RESUME = config.experiment_setting.train.resume
    CHECKPOINT_PATH = config.experiment_setting.train.checkpoint_path
    LOAD_ONLY_MODEL = config.experiment_setting.train.load_only_model
    EPOCHS = config.experiment_setting.train.epochs
    EVAL_ONLY = config.experiment_setting.evaluation.eval_only
    SAVED_MODEL_PATH = config.experiment_setting.evaluation.model_weights_path
    
    # Initialize Dataset and DataLoader
    trainloader, validloader, testloader = get_dataloader(config)
   
    # Initialize Model, Optimizer and Scheduler
    model, optimizer = initialize_model_and_optimizer(config, DEVICE, writer)
    scheduler = initialize_scheduler(optimizer, config)

    # Initialize or Load Checkpoint (If resuming training or just evaluating)
    start_epoch = 0
    if RESUME and CHECKPOINT_PATH:
        print("Resuming training from last checkpoint.")
        CHECKPOINT_PATH = os.path.join(CHECKPOINT_PATH, 'checkpoint.pth')
        model, optimizer, start_epoch = initialize_or_load_checkpoint(model, optimizer, CHECKPOINT_PATH, LOAD_ONLY_MODEL)
    elif EVAL_ONLY and SAVED_MODEL_PATH:
        print("Loading model weights for evaluation.")
        SAVED_MODEL_PATH = os.path.join(SAVED_MODEL_PATH, 'checkpoint.pth')
        model, optimizer, start_epoch = initialize_or_load_checkpoint(model, optimizer, SAVED_MODEL_PATH, LOAD_ONLY_MODEL)
        
    # If evaluation only, directly proceed to the evaluation step and exit    
    if EVAL_ONLY:
        trainer = Trainer(config, model, optimizer, trainloader, testloader, writer)
        eval_metrics = trainer.eval_step(0)  # epoch argument can be 0 since it's not used in evaluation
        print(f"Evaluation loss: {eval_metrics['val_loss']}")
            
        if log_file:
            log_file.write(f"Validation Loss = {eval_metrics['val_loss']}\n")
            if eval_metrics["evaluation_scores"] is not None:
                for metric, scores in eval_metrics['evaluation_scores'].items():
                    log_file.write(f"{metric} = ")
                    for scale, score in scores.items():
                        log_file.write(f"Scale {scale}: {score:.4f}, ")
                    log_file.write("\n")
            log_file.close()        
        return
    
    # Main Training Loop
    best_val_loss = float('inf')
    early_stopping_patience = 30  
    epochs_without_improvement = 0

    for epoch in range(start_epoch, EPOCHS):
        train_metrics, eval_metrics = train_and_eval_model(config, epoch, model, optimizer, scheduler, trainloader, validloader, writer, log_file)

        # Update the best validation loss if current epoch's loss is lower
        if eval_metrics['val_loss'] < best_val_loss:
            best_val_loss = eval_metrics['val_loss']
            epochs_without_improvement = 0
            
            # Save the model checkpoint
            save_checkpoint_path = os.path.join(log_dir, 'checkpoint.pth')
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
            }, save_checkpoint_path)
        else:
            epochs_without_improvement += 1
        
        # # Early stopping condition
        # if epochs_without_improvement >= early_stopping_patience:
        #     print(f"Early stopping triggered: no improvement for {early_stopping_patience} epochs")
        #     break

    if writer:
        writer.close()
    if log_file:
        log_file.close()


if __name__ == "__main__":
    main()
