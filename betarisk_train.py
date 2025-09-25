"""
betarisk_train.py

This script implements the training and finetuning process for a multi-scale probabilistic
model designed for roadway fatality risk prediction. It uses a ResNet50 backbone, multi-scale 
image inputs, and predicts Beta distribution parameters along with classification logits.

Key Features:
- Utilizes custom datasets, augmentations, models, and optimization utilities from the './utils' directory.
- Supports pre-trained weights for the feature extractor (InfoNCE+CLS or ImageNet).
- Employs WassersteinBetaLoss for distribution regression and CrossEntropyLoss for classification.
- Implements mixed-precision training with GradScaler for efficiency.
- Logs training, validation, and testing metrics to TensorBoard.
- Saves model checkpoints based on validation accuracy.

Dependencies:
- torch, torchvision, numpy, tqdm, tensorboard, os, random
- Custom modules: utilities, mydatasets, myaugmentations, mymodels, myopts
"""

import sys
sys.path.append("./utils")

# Custom utility imports
import utilities as UT
import mydatasets as MyDatasets
import myaugmentations as MyAugs
import mymodels as MyModels
import myopts as MyOpts

# Custom loss function import
from wasserstein_beta_loss import WassersteinBetaLoss

# Standard library imports
import os
import numpy as np
import tqdm

# PyTorch imports
import torch 
import torch.nn as nn
import torch.optim as optim

# PyTorch distributed/AMP/scheduler imports
import torch.nn.parallel as parallel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import GradScaler, autocast

# PyTorch data handling imports
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# Set random seeds for reproducibility
import random
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0) # if you using multi-GPU.
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Display GPU information
UT.show_gpu()

# ---------------------------------------------------------------------------- #
#                              Finetune Hyperparameters                        #
# ---------------------------------------------------------------------------- #
epochs = 25
batch_size = 48
num_workers = 8

num_classes = 2
learning_rate = "variable" # Learning rate is set per parameter group in the optimizer
class_weights = [1.25948, 4.85382] # Inverse frequency class weights

debug = False # Flag to indicate if debug mode is enabled

# Set device for training (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Weights for combining different loss components
loss_weights = {"distribution_weight": 10.0, "cls_weight": 1.0}

# Configuration for the Wasserstein Beta Loss function
wasserstein_loss_config = {
    "reduction": "mean",
    "eps": 1e-8,
    "alpha_min": 1e-3,
    "beta_min": 1e-3,
    "weight": None # Class weights for the loss, set to None here to use standard
}

# Learning rates for different parts of the model (feature extractor, conv layers, FC heads)
lr = {"ft_img": 1e-4, "conv_img": 1e-4, "fc_alpha_raw": 2e-2,
       "fc_beta_raw": 2e-2, "fc_classifier": 1e-4}

# Parameters for constructing the target Beta distribution in the dataset
beta_distribution_parameters = {"base_certainty_K": 22.0, "epsilon_beta": 0.08,
                                "min_positive_risk_mean": 0.18, 
                                "min_concentration_positives": 18.0}

best_acc = 0 # To keep track of the best validation accuracy

# ---------------------------------------------------------------------------- #
#                               Pre-training Configuration                     #
# ---------------------------------------------------------------------------- #
pretrain = True # Flag to indicate if InfoNCE+CLS pretrained weights are used
pretrain_method = "InfoNCE+CLS" if pretrain else "ImageNet"
pretrain_ckpt_folder = "ckpts/pre-trained_weights"
pretrain_ckpt_name = "pre-trained_weights_mscm.pth"
pretrain_ckpt_path = os.path.join(pretrain_ckpt_folder, pretrain_ckpt_name)

# ---------------------------------------------------------------------------- #
#                               Finetuning Configuration                       #
# ---------------------------------------------------------------------------- #
trail = "01" # Identifier for the current experiment/trial
optimizer_name = "AdamW"
model_name = f"Res50-FineTune-MultiScale-Probabilistic_PreTrain-{pretrain_method}"

# Construct a unique output name for logs and checkpoints
output_name = f"{model_name}_{optimizer_name}_{str(learning_rate)}_{str(batch_size)}_{str(trail)}"
output_folder = "./"

# Define paths for saving model checkpoints and TensorBoard logs
ckpt_path = os.path.join(output_folder, "ckpts", output_name)
log_path = os.path.join(output_folder, "logs", output_name)

# Create output directories if they don't exist
if not os.path.exists(ckpt_path):
    os.makedirs(ckpt_path)
if not os.path.exists(log_path):
    os.makedirs(log_path)

# Initialize TensorBoard SummaryWriter for logging metrics
writer = SummaryWriter(log_dir=log_path)

# ---------------------------------------------------------------------------- #
#                           Print Critical Hyperparameters                     #
# ---------------------------------------------------------------------------- #
print("********************************")
print("Use CrossEntropyLoss for classification and Wasserstein (w/o class weights) for distribution loss")
print("--------------------------------")
print(f"Pretrain method: {pretrain_method}")
print(f"Pretrain checkpoint path: {pretrain_ckpt_path}")
print(f"Optimizer name: {optimizer_name}")
print(f"Model name: {model_name}")
print(f"Output folder: {output_folder}")
print(f"Output name: {output_name}")
print("--------------------------------")
print(f"Weights: {loss_weights}")
print(f"Learning rate: {lr}")
print(f"Beta distribution parameters: {beta_distribution_parameters}")
print(f"Wasserstein loss config: {wasserstein_loss_config}")
print("--------------------------------")
print("********************************")


# ---------------------------------------------------------------------------- #
#                               Dataset and DataLoader Setup                   #
# ---------------------------------------------------------------------------- #
# Initialize training, validation, and test datasets using MultiScaleProbabilistic
# and base augmentations.
train_dataset = MyDatasets.MultiScaleProbabilistic(basic_transform=MyAugs.base_aug(), 
                                      pos_metadata="train_pos.csv", neg_metadata="train_neg.csv",
                                      base_certainty_K=beta_distribution_parameters["base_certainty_K"],
                                      epsilon_beta=beta_distribution_parameters["epsilon_beta"],
                                      min_positive_risk_mean=beta_distribution_parameters["min_positive_risk_mean"],
                                      min_concentration_positives=beta_distribution_parameters["min_concentration_positives"])

val_dataset = MyDatasets.MultiScaleProbabilistic(basic_transform=MyAugs.base_aug(),
                                     pos_metadata="val_pos.csv", neg_metadata="val_neg.csv", test=True,
                                     base_certainty_K=beta_distribution_parameters["base_certainty_K"],
                                     epsilon_beta=beta_distribution_parameters["epsilon_beta"],
                                     min_positive_risk_mean=beta_distribution_parameters["min_positive_risk_mean"],
                                     min_concentration_positives=beta_distribution_parameters["min_concentration_positives"])

test_dataset = MyDatasets.MultiScaleProbabilistic(basic_transform=MyAugs.base_aug(),
                                      pos_metadata="test_pos.csv", neg_metadata="test_neg.csv", test=True,
                                      base_certainty_K=beta_distribution_parameters["base_certainty_K"],
                                      epsilon_beta=beta_distribution_parameters["epsilon_beta"],
                                      min_positive_risk_mean=beta_distribution_parameters["min_positive_risk_mean"],
                                      min_concentration_positives=beta_distribution_parameters["min_concentration_positives"])

# Initialize DataLoaders for training, validation, and testing
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, 
                              num_workers=num_workers, shuffle=True, drop_last=True)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)


# ---------------------------------------------------------------------------- #
#                               Model Initialization                           #
# ---------------------------------------------------------------------------- #
# Load pretrained weights for the feature extractor (ResNet50 backbone)
state_dict = torch.load(pretrain_ckpt_path, weights_only=True, map_location=torch.device('cpu'))

# Remove "module." prefix from state_dict keys if present (common with DataParallel)
key_map = {}
for key in state_dict.keys():
    new_key = key.replace("module.", "")
    key_map[key] = new_key

renamed_state_dict = {}
for key, value in state_dict.items():
    renamed_state_dict[key_map[key]] = value
    
# Initialize the multi-scale probabilistic model with the pre-trained feature extractor
model = MyModels.ResNet50_MultiScale_Probabilistic(pretrain_feature_extractor=renamed_state_dict)

# ---------------------------------------------------------------------------- #
#                          Optimizer and Loss Function Setup                   #
# ---------------------------------------------------------------------------- #
# Configure AdamW optimizer with different learning rates for distinct model parameter groups
optimizer = torch.optim.AdamW([
    {'params': model.ft_img.parameters(), 'lr': lr["ft_img"]},
    {'params': model.conv_img.parameters(), 'lr': lr["conv_img"]},
    {'params': model.fc_alpha_raw.parameters(), 'lr': lr["fc_alpha_raw"]},
    {'params': model.fc_beta_raw.parameters(), 'lr': lr["fc_beta_raw"]},
    {'params': model.fc_classifier.parameters(), 'lr': lr["fc_classifier"]}
], weight_decay=1e-4)

# Wrap the model with DataParallel for multi-GPU training
model = parallel.DataParallel(model)
model.to(device)

# Move class weights to the device for CrossEntropyLoss
class_weights = torch.tensor(class_weights).clone().detach().to(device) 
# Initialize CrossEntropyLoss for classification
criterion_cls = nn.CrossEntropyLoss(class_weights)
# Initialize WassersteinBetaLoss for probabilistic distribution regression
criterion_distribution = WassersteinBetaLoss(**wasserstein_loss_config)

# Initialize Cosine Annealing with Warm Restarts scheduler for learning rate adjustment
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
# Initialize GradScaler for mixed precision training
scaler = GradScaler("cuda")

# ---------------------------------------------------------------------------- #
#                                Training Loop                                 #
# ---------------------------------------------------------------------------- #
for epoch in tqdm.tqdm(range(epochs)):
    # Run a single training epoch
    model, epoch_loss, train_acc, _, _ = MyOpts.train_test_multiscale_probabilistic(
                                            epoch, train_dataloader, model, criterion_distribution, criterion_cls, 
                                            optimizer, scheduler, scaler, device, writer, "Train",
                                            distribution_weight=loss_weights["distribution_weight"], 
                                            cls_weight=loss_weights["cls_weight"], debug=debug)

    # Run a single validation epoch
    _, _, val_acc, _, _ = MyOpts.train_test_multiscale_probabilistic(
                                            epoch, val_dataloader, model, criterion_distribution, criterion_cls, 
                                            optimizer, scheduler, scaler, device, writer, "Val",
                                            distribution_weight=loss_weights["distribution_weight"], 
                                            cls_weight=loss_weights["cls_weight"], debug=debug)

    # Run a single test epoch
    _, _, test_acc, _, _ = MyOpts.train_test_multiscale_probabilistic(
                                            epoch, test_dataloader, model, criterion_distribution, criterion_cls, 
                                            optimizer, scheduler, scaler, device, writer, "Test",
                                            distribution_weight=loss_weights["distribution_weight"], 
                                            cls_weight=loss_weights["cls_weight"], debug=debug)

    # Save model checkpoint
    torch.save(model.module.state_dict(), os.path.join(ckpt_path,f'{epoch}_{train_acc}_{val_acc}_{test_acc}.pth'))
    
    # Print epoch summary
    print("Epoch: %d\t Train [loss/acc]: [%.4f/%.4f]\t Val/Test Acc: %.4f/%.4f" 
          %(epoch, epoch_loss, train_acc, val_acc, test_acc))
    
    # Update best accuracy and print if current model is the best
    if best_acc < val_acc:
        best_acc = val_acc
        print("\t\tCurrent best model\n")