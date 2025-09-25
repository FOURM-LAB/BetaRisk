import utilities as UT
import torch 
import tqdm
from torch.amp import GradScaler, autocast
import numpy as np
import torch.nn.functional as F


def train_test_multiscale_probabilistic(epoch, dataloader, model, distribution_criterion, cls_criterion, 
                                        optimizer, scheduler, scaler, device, writer, mode="Train", 
                                        debug=False, distribution_weight=5.0, cls_weight=1.0, 
                                        distribution_regularization=False):
    """
    Training/testing function for multi-scale probabilistic models with Beta distribution output.
    
    Args:
        epoch: Current epoch number
        dataloader: Data loader for multi-scale probabilistic data
        model: Probabilistic model (ResNet50_MultiScale_Probabilistic)
        distribution_criterion: Distribution loss function (e.g., WassersteinBetaLoss)
        cls_criterion: Classification loss function (e.g., CrossEntropyLoss)
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        scaler: Gradient scaler for mixed precision
        device: Device (CPU/GPU)
        writer: TensorBoard writer
        mode: "Train", "Val", or "Test"
        debug: Debug mode flag
        distribution_weight: Weight for the distribution loss component.
        cls_weight: Weight for the classification loss component.
        distribution_regularization: Flag to enable risk distribution regularization.
    """
    
    def risk_distribution_loss(pred_risk, target_risk):
        """Penalize systematic bias in risk predictions"""
        pred_mean = pred_risk.mean()
        target_mean = target_risk.mean()
        return F.mse_loss(pred_mean, target_mean)

    assert mode in ["Train", "Val", "Test"], f"Invalid mode: {mode}. Mode must be 'Train', 'Val', or 'Test'."
    
    model.train() if mode == "Train" else model.eval()
        
    running_loss, running_distribution_loss, running_cls_loss = 0.0, 0.0, 0.0
    total, correct = 0, 0

    image_key = "aug" if mode == "Train" else "normal"

    with torch.set_grad_enabled(mode == "Train"):
        for i, batch in enumerate(dataloader):            
            labels = batch["label"].to(device)
            alpha_target = batch["alpha_target"].to(device)
            beta_target = batch["beta_target"].to(device)
            
            # Handle multi-scale image list
            images = [img.to(device) for img in batch[image_key]]

            if mode == "Train":
                optimizer.zero_grad()
                with autocast("cuda"):
                    outputs = model(images)
                    cls_loss = cls_criterion(outputs['logits'], labels)
                
                distribution_loss = distribution_criterion(outputs['alpha'], outputs['beta'], alpha_target, beta_target, labels)
                
                total_loss = distribution_weight * distribution_loss + cls_weight * cls_loss 
                if distribution_regularization:
                    true_risk_mean = (alpha_target / (alpha_target + beta_target)).mean()
                    total_loss += 0.5 * F.mse_loss(outputs['risk_score'].mean(), true_risk_mean)

                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
            
            else: # "Val" or "Test"
                with torch.no_grad():
                    outputs = model(images)
                    cls_loss = cls_criterion(outputs['logits'], labels)
                    distribution_loss = distribution_criterion(outputs['alpha'], outputs['beta'], alpha_target, beta_target, labels)
                    
                    total_loss = distribution_weight * distribution_loss + cls_weight * cls_loss
                    if distribution_regularization:
                        true_risk_mean = (alpha_target / (alpha_target + beta_target)).mean()
                        total_loss += 0.5 * F.mse_loss(outputs['risk_score'].mean(), true_risk_mean)

            # Update running losses and metrics
            batch_size = images[0].size(0)
            running_loss += total_loss.item() * batch_size
            running_distribution_loss += distribution_loss.item() * batch_size
            running_cls_loss += cls_loss.item() * batch_size
        
            total += labels.size(0)
            _, predicted = torch.max(outputs['logits'].detach(), 1)
            correct += (predicted == labels).sum().item()
        
            # --- Logging ---
            if writer:
                global_step = epoch * len(dataloader) + i
                writer.add_scalar(f'{mode}/Running/Total_Loss', total_loss.item(), global_step)
                writer.add_scalar(f'{mode}/Running/Distribution_Loss', distribution_loss.item(), global_step)
                writer.add_scalar(f'{mode}/Running/Cls_Loss', cls_loss.item(), global_step)
                writer.add_scalar(f'{mode}/Running/Acc', correct / total, global_step)
                
                if mode == "Train":
                    writer.add_scalar('Train/Running/LR', optimizer.param_groups[0]['lr'], global_step)
                    
                # Log risk and distribution stats
                true_risk = alpha_target / (alpha_target + beta_target)
                predicted_risk = outputs['risk_score']
                risk_correlation = torch.corrcoef(torch.stack([predicted_risk, true_risk]))[0, 1].item() if total > 1 else 0
                
                writer.add_scalar(f'{mode}/Running/Mean_Alpha_Pred', outputs['alpha'].mean().item(), global_step)
                writer.add_scalar(f'{mode}/Running/Mean_Beta_Pred', outputs['beta'].mean().item(), global_step)
                writer.add_scalar(f'{mode}/Running/Mean_Risk_Pred', predicted_risk.mean().item(), global_step)
                writer.add_scalar(f'{mode}/Running/Risk_Correlation', risk_correlation, global_step)

            if debug:
                break
            
    # Calculate and log epoch metrics
    epoch_total_loss = running_loss / len(dataloader.dataset)
    epoch_distribution_loss = running_distribution_loss / len(dataloader.dataset)
    epoch_cls_loss = running_cls_loss / len(dataloader.dataset)
    accuracy_overall = correct / total
    
    if writer:
        writer.add_scalar(f'{mode}/Epoch/Total_Loss', epoch_total_loss, epoch)
        writer.add_scalar(f'{mode}/Epoch/Distribution_Loss', epoch_distribution_loss, epoch)
        writer.add_scalar(f'{mode}/Epoch/Cls_Loss', epoch_cls_loss, epoch)
        writer.add_scalar(f'{mode}/Epoch/Acc', accuracy_overall, epoch)

    if debug:
        print(f"Epoch {epoch} {mode} total loss: {epoch_total_loss:.4f}, dist loss: {epoch_distribution_loss:.4f}, cls loss: {epoch_cls_loss:.4f}, acc: {accuracy_overall:.4f}")

    return model, epoch_total_loss, accuracy_overall, epoch_distribution_loss, epoch_cls_loss