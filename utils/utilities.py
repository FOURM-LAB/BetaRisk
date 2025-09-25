import torch
from torchvision import transforms

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix
)
import seaborn as sns

from sklearn.calibration import calibration_curve

from scipy.stats import beta



def show_gpu():
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")

        # Get the names of the available GPUs
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print("Available GPUs:")
        for gpu_id, gpu_name in enumerate(gpu_names):
            gpu_device = torch.device(f'cuda:{gpu_id}')
            gpu_memory_allocated = torch.cuda.max_memory_allocated(gpu_device) / 1024**3  # Convert bytes to GBs
            gpu_memory_allocated_cached = torch.cuda.memory_allocated(gpu_device) / 1024**3  # Convert bytes to GBs
            gpu_memory_allocated_diff = gpu_memory_allocated_cached - gpu_memory_allocated
            gpu_memory_allocated_diff = round(gpu_memory_allocated_diff, 2)
            print(f'GPU {gpu_id}: {gpu_name}: Memory Allocated: {gpu_memory_allocated:.2f} GB, Memory Allocated (Cached): {gpu_memory_allocated_cached:.2f} GB, Difference: {gpu_memory_allocated_diff:.2f} GB')
    else:
        print("CUDA is not available on this system.") 
                

def read_csv(data_file_path):
    data = []
    with open(data_file_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        data = np.asarray(data)
    return data

# This function is used to denormlize the images based on MEAN and STD
def img_denorm(img, mean=np.asarray([0.485,0.456,0.406]), 
               std=np.asarray([0.229,0.224,0.225]),
               transpose = True):
    #for ImageNet the mean and std are:
    mean = np.asarray(mean)
    std = np.asarray(std)
    denormalize = transforms.Normalize((-1 * mean / std), (1.0 / std))

    res = img.squeeze(0)
    res = denormalize(res)

    #Image needs to be clipped since the denormalize function will map some
    #values below 0 and above 1
    res = torch.clamp(res, 0, 1)
    
    if(transpose):
        res = res.numpy()
        res = np.transpose(res, (1, 2, 0))
    
    return res


def img_transpose(img):  
    res = img 
    res = res.numpy()
    res = np.transpose(res, (1, 2, 0))
    
    return res


def check_gpu_usage():
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


def save_evaluation_results(all_labels_tensor, all_outputs_tensor, save_dir, name="evaluation_results.pth"):
    """
    Saves the collected true labels and model outputs to a specified directory.

    Args:
        all_labels_tensor (torch.Tensor): Tensor containing all true labels from evaluation.
        all_outputs_tensor (torch.Tensor): Tensor containing all model outputs (logits) from evaluation.
        save_dir: The directory for saving the evaluation results.
    """

    # Define the full file paths for your tensors (Saving them together in a single dictionary)
    combined_data_file_path = os.path.join(save_dir, name)

    # Save the tensors in a dictionary
    try:
        torch.save({
            'labels': all_labels_tensor,
            'outputs': all_outputs_tensor
        }, combined_data_file_path)
        print(f"Combined evaluation results saved to: {combined_data_file_path}")
    except Exception as e:
        print(f"Error saving tensors to {combined_data_file_path}: {e}")

def save_evaluation_results_probabilistic(all_labels_tensor, all_alpha_target_tensor, all_beta_target_tensor, 
                                          all_alpha_pred_tensor, all_beta_pred_tensor, all_logits_tensor, 
                                          all_risk_score_tensor, save_dir, name="evaluation_results.pth"):
    """
    Saves the collected true labels and model outputs to a specified directory.

    Args:
        all_labels_tensor (torch.Tensor): Tensor containing all true labels from evaluation.
        all_outputs_tensor (torch.Tensor): Tensor containing all model outputs (logits) from evaluation.
        save_dir: The directory for saving the evaluation results.
    """

    # Define the full file paths for your tensors (Saving them together in a single dictionary)
    combined_data_file_path = os.path.join(save_dir, name)

    # Save the tensors in a dictionary
    try:
        torch.save({
            'labels': all_labels_tensor,
            'alpha_target': all_alpha_target_tensor,
            'beta_target': all_beta_target_tensor,
            'alpha_pred': all_alpha_pred_tensor,
            'beta_pred': all_beta_pred_tensor,
            'logits': all_logits_tensor,
            'risk_score': all_risk_score_tensor
        }, combined_data_file_path)
        print(f"Combined evaluation results saved to: {combined_data_file_path}")
    except Exception as e:
        print(f"Error saving tensors to {combined_data_file_path}: {e}")        


def plot_brier_net_comparison(results, ax, n_bins=15, main_title="Model Comparison: Net Calibration Error", fontsize=12):
    """
    Plots only the net calibration error (Reliability - Resolution) for model comparison.
    Since Uncertainty is identical for all models on the same dataset, we can ignore it.
    
    Lower bars = Better models
    Negative values = Model skill (Resolution) outweighs calibration error (Reliability)
    """
    model_names = list(results.keys())
    
    # Calculate components for each model
    net_errors = []
    decompositions = {}
    
    for model_name, (y_true, y_prob) in results.items():
        reliability, resolution, uncertainty = compute_brier_decomposition(y_true, y_prob, n_bins)
        net_error = reliability - resolution
        net_errors.append(net_error)
        
        decompositions[model_name] = {
            'Reliability': reliability,
            'Resolution': resolution, 
            'Net_Error': net_error,
            'Uncertainty': uncertainty
        }
    
    # Print summary
    print(f"\n--- NET CALIBRATION ERROR COMPARISON ---")
    print(f"Net Error = Reliability - Resolution (lower is better)")
    for model_name, data in decompositions.items():
        net = data['Net_Error']
        print(f"  {model_name}: {net:.6f} (Rel: {data['Reliability']:.6f} - Res: {data['Resolution']:.6f})")
    
    # Clear the axis
    ax.clear()
    
    # Create the plot
    bar_width = 0.6
    index = np.arange(len(model_names))
    
    # Fixed colors for the three bars
    colors = ['#1f77b4', '#d62728', '#2ca02c']  # Blue, Red, Green
    
    # Create bars with labels for legend
    bars = []
    for i, (model_name, net_error) in enumerate(zip(model_names, net_errors)):
        bar = ax.bar(index[i], net_error, bar_width, 
                    color=colors[i], alpha=0.7, edgecolor='black', linewidth=1,
                    label=model_name)
        bars.append(bar)
    
    # Add a horizontal line at y=0 for reference
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.7, linewidth=1)
    
    # Labels and formatting
    ax.set_ylabel('Net Calibration Error\n(Reliability - Resolution)', fontsize=fontsize)
    ax.set_title(main_title)
    # ax.set_xticks(index)
    # ax.set_xticklabels(model_names, rotation=0, fontsize=fontsize)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Add legend
    ax.legend(fontsize=fontsize-1, loc='lower right')
    
    # Add value labels on the opposite side of y=0
    for i, (bar, net_error) in enumerate(zip(bars, net_errors)):
        height = bar[0].get_height()  # Need [0] because bar is now a container
        # For positive bars, put text below y=0
        # For negative bars, put text above y=0
        if height >= 0:
            label_y = -0.0004  # Reduced offset below y=0
            va = 'top'
        else:
            label_y = 0.0004  # Reduced offset above y=0
            va = 'bottom'
        ax.text(bar[0].get_x() + bar[0].get_width()/2., label_y,
                f'{net_error:.4f}', ha='center', va=va,
                fontweight='bold', fontsize=fontsize,
                color='black')
    
    # Add explanatory text - moved to top right
    ax.text(0.98, 0.98, 
            'Lower values = Better models\n' +
            'Negative values = Net skill (Resolution > Reliability)\n' +
            'Positive values = Net error (Reliability > Resolution)',
            transform=ax.transAxes, fontsize=fontsize-1, verticalalignment='top',
            horizontalalignment='right',  # Align text to the right
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    print(f"--- Best model: {model_names[np.argmin(net_errors)]} (lowest net error) ---\n")
    return ax

def plot_brier_decomposition(results, ax, n_bins=15, main_title="Brier Score Decomposition"):
    """
    Plots a stacked bar chart of the Brier score's decomposed components on a given axis.
    """
    model_names = list(results.keys())
    
    # Calculate components for each model
    decompositions = {}
    for model_name, (y_true, y_prob) in results.items():
        reliability, resolution, uncertainty = compute_brier_decomposition(y_true, y_prob, n_bins)
        decompositions[model_name] = {
            'Brier Score': reliability - resolution + uncertainty,
            'Reliability': reliability, 
            'Resolution': resolution, 
            'Uncertainty': uncertainty
        }
    
    # --- FINAL DEBUG STEP: Print the values right before plotting ---
    print("\n--- FINAL DEBUG: Data passed to plotting function ---")
    for model_name, data in decompositions.items():
        print(f"  Model: {model_name}, Uncertainty: {data['Uncertainty']:.6f}, Reliability: {data['Reliability']:.6f}, Resolution: {data['Resolution']:.6f}")

    # Prepare data for plotting - EXPLICIT EXTRACTION
    uncertainty_scores = []
    reliability_scores = []
    resolution_scores = []
    
    for model_name in model_names:
        data = decompositions[model_name]
        uncertainty_scores.append(data['Uncertainty'])
        reliability_scores.append(data['Reliability'])
        # CRITICAL FIX: Use absolute value of resolution for visual height, but keep track of sign
        resolution_scores.append(abs(data['Resolution']))  # Always positive for upward bars
    
    # --- ADDITIONAL DEBUG: Print the exact arrays being plotted ---
    print(f"\n--- PLOTTING ARRAYS DEBUG ---")
    print(f"  Model names: {model_names}")
    print(f"  Uncertainty (gray bars): {[f'{s:.6f}' for s in uncertainty_scores]}")
    print(f"  Reliability (red bars): {[f'{s:.6f}' for s in reliability_scores]}")
    print(f"  |Resolution| (green bars): {[f'{s:.6f}' for s in resolution_scores]}")

    # Clear the axis completely to avoid any caching issues
    ax.clear()
    
    # Create the plot with explicit parameters
    bar_width = 0.6
    index = np.arange(len(model_names))
    
    # Plot each component separately with explicit debugging
    print(f"\n--- MATPLOTLIB PLOTTING DEBUG ---")
    
    # 1. Uncertainty bars (gray, bottom layer)
    bars1 = ax.bar(index, uncertainty_scores, bar_width, 
                   label='Uncertainty', color='gray', alpha=0.8)
    print(f"  Plotted Uncertainty bars with heights: {[f'{bar.get_height():.6f}' for bar in bars1]}")
    
    # 2. Reliability bars (red, stacked on uncertainty)
    bars2 = ax.bar(index, reliability_scores, bar_width, 
                   bottom=uncertainty_scores, 
                   label='Reliability (Error)', color='#d62728', alpha=0.8)
    print(f"  Plotted Reliability bars with heights: {[f'{bar.get_height():.6f}' for bar in bars2]}")
    print(f"  Reliability bars bottoms: {[f'{s:.6f}' for s in uncertainty_scores]}")
    
    # 3. Resolution bars (green, but we need to handle the direction properly)
    # Since Resolution reduces Brier score (negative contribution), we stack it BELOW uncertainty
    # But we want it to appear as extending the total bar height for visual clarity
    bottom_for_resolution = [u + r for u, r in zip(uncertainty_scores, reliability_scores)]
    bars3 = ax.bar(index, resolution_scores, bar_width, 
                   bottom=bottom_for_resolution,
                   label='Resolution (Skill)', color='#2ca02c', alpha=0.8)
    print(f"  Plotted Resolution bars with heights: {[f'{bar.get_height():.6f}' for bar in bars3]}")
    print(f"  Resolution bars bottoms: {[f'{b:.6f}' for b in bottom_for_resolution]}")
    
    # Set labels and formatting
    ax.set_ylabel('Score')
    ax.set_title(main_title)
    ax.set_xticks(index)
    ax.set_xticklabels(model_names)
    ax.legend(loc='best')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels for total Brier scores
    for i, model_name in enumerate(model_names):
        brier_score = decompositions[model_name]['Brier Score']
        total_height = uncertainty_scores[i] + reliability_scores[i] + resolution_scores[i]
        ax.text(index[i], total_height + 0.005, f'Brier: {brier_score:.4f}', 
                ha='center', va='bottom', fontweight='bold')
    
    # Add explanatory text
    # ax.text(0.02, 0.98, 
    #         'Brier Score = Uncertainty + Reliability - Resolution\n' +
    #         'Lower is better. Resolution (green) represents model skill.',
    #         transform=ax.transAxes, fontsize=9, verticalalignment='top',
    #         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    print(f"--- END PLOTTING DEBUG ---\n")
    return ax

def plot_ece_by_bins(results, bins_to_test=None, main_title="ECE vs. Number of Bins"):
    """
    Computes and plots the ECE for multiple models across a range of bin numbers.
    """
    if bins_to_test is None:
        bins_to_test = range(5, 51, 5)

    plt.figure(figsize=(12, 8))
    
    colors = ['#d62728', '#1f77b4', '#2ca02c'] # Red, Blue, Green
    
    for i, (model_name, (y_true, y_prob)) in enumerate(results.items()):
        color = colors[i % len(colors)]
        ece_scores = []
        for n_bins in bins_to_test:
            # Revert back to the original function
            ece = compute_ece(y_true, y_prob, n_bins=n_bins)
            ece_scores.append(ece)
        
        plt.plot(bins_to_test, ece_scores, marker='o', linestyle='-',
                 label=model_name, color=color)

    plt.xlabel("Number of Bins")
    plt.ylabel("Expected Calibration Error (ECE)")
    plt.title(main_title)
    plt.xticks(bins_to_test)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def plot_probability_distributions(results, n_bins=50, main_title="Model Probability Distributions"):
    """
    Plots histograms of the predicted probabilities for multiple models.

    Args:
        results (dict): A dictionary where keys are model names and values are
                        tuples of (y_true, y_pred), where y_pred are probabilities.
        n_bins (int): Number of bins for the histogram.
        main_title (str): The main title for the figure.
    """
    num_models = len(results)
    fig, axes = plt.subplots(num_models, 1, figsize=(10, 4 * num_models), sharex=True)
    if num_models == 1:
        axes = [axes]

    colors = ['#d62728', '#1f77b4', '#2ca02c'] # Red, Blue, Green to match your plots

    for i, (ax, (model_name, (y_true, y_prob))) in enumerate(zip(axes, results.items())):
        color = colors[i % len(colors)]
        
        # Separate probabilities for true positive and true negative classes
        pos_probs = y_prob[y_true == 1]
        neg_probs = y_prob[y_true == 0]

        ax.hist([neg_probs, pos_probs], bins=n_bins, range=(0, 1),
                stacked=True, color=[color, '#ff7f0e'], # Base color and orange
                label=[f'{model_name} (True Negative)', f'{model_name} (True Positive)'],
                alpha=0.7)

        ax.set_title(f"'{model_name}' Predictions")
        ax.set_ylabel("Number of Samples")
        ax.legend()
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    axes[-1].set_xlabel("Predicted Probability")
    fig.suptitle(main_title, fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def compute_brier_decomposition(y_true, y_prob, n_bins=15):
    """
    Decomposes the Brier score into its components: Reliability, Resolution, and Uncertainty.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    print(f"\n=== BRIER DECOMPOSITION DEBUG ===")
    print(f"Input shapes: y_true={y_true.shape}, y_prob={y_prob.shape}")
    print(f"y_true range: [{y_true.min()}, {y_true.max()}], unique values: {np.unique(y_true)}")
    print(f"y_prob range: [{y_prob.min():.4f}, {y_prob.max():.4f}]")

    # Bin predictions
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_boundaries[1:-1])
    
    print(f"Bin boundaries: {[f'{b:.3f}' for b in bin_boundaries]}")
    unique_bins, counts = np.unique(bin_indices, return_counts=True)
    print(f"Samples per bin: {dict(zip(unique_bins, counts))}")

    # Overall prevalence of the positive class
    prevalence = np.mean(y_true)
    print(f"Overall prevalence: {prevalence:.6f}")
    
    # Calculate uncertainty component
    uncertainty = prevalence * (1.0 - prevalence)
    print(f"Uncertainty = {prevalence:.6f} * {1.0 - prevalence:.6f} = {uncertainty:.6f}")

    reliability = 0.0
    resolution = 0.0
    
    print(f"\nBin-by-bin calculations:")
    for i in range(n_bins):
        in_bin = bin_indices == i
        n_in_bin = np.sum(in_bin)

        if n_in_bin > 0:
            # Observed frequency in bin
            obs_freq = np.mean(y_true[in_bin])
            # Average predicted probability in bin
            pred_prob = np.mean(y_prob[in_bin])
            
            reliability_contrib = n_in_bin * (pred_prob - obs_freq)**2
            resolution_contrib = n_in_bin * (obs_freq - prevalence)**2
            
            reliability += reliability_contrib
            resolution += resolution_contrib
            
            print(f"  Bin {i}: n={n_in_bin}, obs_freq={obs_freq:.4f}, pred_prob={pred_prob:.4f}")
            print(f"    Reliability contrib: {n_in_bin} * ({pred_prob:.4f} - {obs_freq:.4f})^2 = {reliability_contrib:.6f}")
            print(f"    Resolution contrib: {n_in_bin} * ({obs_freq:.4f} - {prevalence:.4f})^2 = {resolution_contrib:.6f}")

    reliability /= len(y_prob)
    resolution /= len(y_prob)
    
    print(f"\nFinal components:")
    print(f"  Reliability (before normalization): {reliability * len(y_prob):.6f}")
    print(f"  Reliability (after normalization): {reliability:.6f}")
    print(f"  Resolution (before normalization): {resolution * len(y_prob):.6f}")
    print(f"  Resolution (after normalization): {resolution:.6f}")
    print(f"  Uncertainty: {uncertainty:.6f}")
    
    brier_score = reliability - resolution + uncertainty
    print(f"  Brier Score = {reliability:.6f} - {resolution:.6f} + {uncertainty:.6f} = {brier_score:.6f}")
    
    # Verify with direct Brier calculation
    direct_brier = np.mean((y_prob - y_true)**2)
    print(f"  Direct Brier calculation: {direct_brier:.6f}")
    print(f"  Difference: {abs(brier_score - direct_brier):.8f}")
    print(f"=== END BRIER DECOMPOSITION DEBUG ===\n")
    
    # Brier Score = Reliability - Resolution + Uncertainty
    return reliability, resolution, uncertainty

def compute_brier_score(y_true, y_prob):
    """Computes the Brier Score."""
    return np.mean((y_prob - y_true)**2)

def compute_ece_debug(y_true, y_prob, n_bins=15, model_name=""):
    """
    A debug version of compute_ece that prints its internal steps.
    """
    print(f"\n--- DEBUG: ECE for '{model_name}' with n_bins={n_bins} ---")
    
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Print initial stats about the probability data
    print(f"  y_prob stats -> shape: {y_prob.shape}, min: {y_prob.min():.4f}, max: {y_prob.max():.4f}, #unique: {np.unique(y_prob).size}")

    # Create and show the bin boundaries
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    print(f"  Bin Boundaries: {[f'{b:.2f}' for b in bin_boundaries]}")

    # Digitize the predictions and show the result
    bin_indices = np.digitize(y_prob, bin_boundaries[1:-1])
    unique_bins, counts = np.unique(bin_indices, return_counts=True)
    print(f"  Binning results -> unique bin indices found: {unique_bins}")
    print(f"  Binning results -> counts per bin: {counts}")

    ece = 0.0
    for i in range(n_bins):
        in_bin = bin_indices == i
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            acc_in_bin = np.mean(y_true[in_bin])
            conf_in_bin = np.mean(y_prob[in_bin])
            error_contribution = np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            ece += error_contribution
            
            # Print the calculation for each non-empty bin
            print(f"  [Bin {i:02d}]: Accuracy={acc_in_bin:.4f}, Confidence={conf_in_bin:.4f}, Proportion={prop_in_bin:.4f}, ECE Contrib={error_contribution:.4f}")
    
    print(f"  ---> FINAL ECE for {n_bins} bins: {ece:.4f}")
    return ece


def compute_ece(y_true, y_prob, n_bins=15):
    """
    Computes the Expected Calibration Error (ECE).

    Args:
        y_true (np.array): True binary labels (0 or 1).
        y_prob (np.array): Predicted probabilities for the positive class.
        n_bins (int): The number of bins to partition the probabilities into.

    Returns:
        float: The Expected Calibration Error.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Bin predictions by their confidence
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    # Get the bin index for each prediction. Bins are 0-indexed.
    bin_indices = np.digitize(y_prob, bin_boundaries[1:-1])

    ece = 0.0
    for i in range(n_bins):
        # Find samples that fall into the i-th bin
        in_bin = bin_indices == i
        prop_in_bin = np.mean(in_bin)

        if prop_in_bin > 0:
            # Accuracy of the bin (fraction of positives)
            acc_in_bin = np.mean(y_true[in_bin])
            # Average confidence of the bin
            conf_in_bin = np.mean(y_prob[in_bin])
            
            # Add to the weighted ECE
            ece += np.abs(acc_in_bin - conf_in_bin) * prop_in_bin
            
    return ece


def compute_metrics(true_labels_tensor, model_outputs_tensor, threshold=0.5,
                    plot_title_prefix="Model Evaluation", show_plot=True, apply_sigmoid=True, n_bins=15):
    """
    Calculates and prints various classification metrics (Accuracy, F1, Precision, Recall, AUC, Brier, ECE),
    and plots ROC and Precision-Recall curves.
    """
    # Ensure tensors are on CPU and converted to numpy arrays
    true_labels_np = true_labels_tensor.cpu().numpy()
    model_outputs_np = model_outputs_tensor.cpu().numpy()

    if apply_sigmoid:
        if model_outputs_np.ndim == 1 or (model_outputs_np.ndim == 2 and model_outputs_np.shape[1] == 1):
            probabilities_np = torch.sigmoid(torch.from_numpy(model_outputs_np.flatten())).numpy()
            predicted_labels_np = (probabilities_np >= threshold).astype(int)
        elif model_outputs_np.ndim == 2 and model_outputs_np.shape[1] > 1:
            predicted_labels_np = model_outputs_np.argmax(axis=1)
            probabilities_np = torch.softmax(torch.from_numpy(model_outputs_np), dim=1)[:, 1].numpy()
        else:
            raise ValueError("model_outputs_tensor has an unexpected shape for classification.")
    else:
        probabilities_np = model_outputs_np
        predicted_labels_np = (probabilities_np >= threshold).astype(int)

    print(f"\n--- {plot_title_prefix} Metrics ---")
    acc = accuracy_score(true_labels_np, predicted_labels_np)
    f1 = f1_score(true_labels_np, predicted_labels_np)
    prec = precision_score(true_labels_np, predicted_labels_np)
    recall = recall_score(true_labels_np, predicted_labels_np)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"Precision (at threshold {threshold}): {prec:.4f}")
    print(f"Recall (at threshold {threshold}): {recall:.4f}")

    try:
        auc_roc = roc_auc_score(true_labels_np, probabilities_np)
        print(f"AUC-ROC: {auc_roc:.4f}")
    except ValueError as e:
        print(f"Could not calculate AUC-ROC: {e}")

    try:
        auc_pr = average_precision_score(true_labels_np, probabilities_np)
        print(f"AUC-PR: {auc_pr:.4f}")
    except ValueError as e:
        print(f"Could not calculate AUC-PR: {e}")

    # --- Calibration Metrics ---
    try:
        brier = compute_brier_score(true_labels_np, probabilities_np)
        print(f"Brier Score (lower is better): {brier:.4f}")
    except Exception as e:
        print(f"Could not calculate Brier Score: {e}")

    try:
        ece = compute_ece(true_labels_np, probabilities_np, n_bins=n_bins)
        print(f"ECE (Expected Calibration Error): {ece:.4f}")
    except Exception as e:
        print(f"Could not calculate ECE: {e}")

    if show_plot:
        fig, axes = plt.subplots(1, 2, figsize=(11, 6))
        fpr, tpr, _ = roc_curve(true_labels_np, probabilities_np)
        axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc_roc:.2f})')
        axes[0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].set_title(f'{plot_title_prefix}\nReceiver Operating Characteristic (ROC) Curve')
        axes[0].legend(loc="lower right")
        axes[0].grid(True)
        precision_points, recall_points, _ = precision_recall_curve(true_labels_np, probabilities_np)
        axes[1].plot(recall_points, precision_points, color='blue', lw=2, label=f'PR curve (area = {auc_pr:.2f})')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        axes[1].set_title(f'{plot_title_prefix}\nPrecision-Recall Curve')
        axes[1].legend(loc="lower left")
        axes[1].grid(True)
        plt.tight_layout()
        plt.show()


def plot_reliability_diagram(true_labels_tensor, all_outputs_tensor, 
                             n_bins=10, figsize=(5, 5), facecolor="green", model_name="MultiScale", apply_sigmoid=True, ax=None, fontsize=12):
    """
    Draws a reliability diagram styled as a bar chart with no gaps between bins.
    The bins are evenly spread across the x-axis.

    Args:
        true_labels_tensor (torch.Tensor): Tensor of true binary labels (0 or 1).
        all_outputs_tensor (torch.Tensor): Tensor of model outputs (logits or probabilities).
        n_bins (int): Number of bins to use for the calibration curve.
        model_name (str): The name to display in the legend for the model's bars.
        apply_sigmoid (bool): If True, applies sigmoid to `all_outputs_tensor`. Set to False if inputs are already probabilities.
        ax (matplotlib.axes.Axes, optional): The axes object to plot on. If None, a new figure and axes are created.
    """

    # Ensure tensors are on CPU and converted to numpy arrays for SKlearn.
    true_labels_np = true_labels_tensor.cpu().numpy()
    outputs_cpu_tensor = all_outputs_tensor.cpu() 

    predicted_probabilities_tensor = outputs_cpu_tensor
    if apply_sigmoid:
        if outputs_cpu_tensor.ndim == 1 or (outputs_cpu_tensor.ndim == 2 and outputs_cpu_tensor.shape[1] == 1):
            predicted_probabilities_tensor = torch.sigmoid(outputs_cpu_tensor.flatten())
        elif outputs_cpu_tensor.ndim == 2 and outputs_cpu_tensor.shape[1] > 1:
            predicted_probabilities_tensor = torch.softmax(outputs_cpu_tensor, dim=1)[:, 1]
        else:
            raise ValueError("all_outputs_tensor has an unexpected shape for classification.")
    
    predicted_probabilities_np = predicted_probabilities_tensor.numpy()

    # Calculate actual positive fraction for each bin.
    fraction_of_positives, mean_predicted_value = calibration_curve(
        true_labels_np, predicted_probabilities_np, n_bins=n_bins, strategy='uniform'
    )

    # --- Plotting ---
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    # Plot the perfect calibration line (dashed blue line)
    ax.plot([0, 1], [0, 1], "k:", label="Perfectly calibrated")

    # Plot the model's calibration curve as a bar chart.
    # The width of each bar is 1/n_bins, and there are no gaps.
    bar_centers = np.linspace(0, 1, n_bins + 1)[:-1] + (1 / (2 * n_bins))
    ax.bar(bar_centers, fraction_of_positives, width=1/n_bins, 
           edgecolor="black", color=facecolor, alpha=0.7, 
           label=f'{model_name} (ECE={compute_ece(true_labels_tensor, predicted_probabilities_tensor, n_bins):.4f})'
        #    label=f'{model_name}'
)
    
    # ax.set_xlabel("Mean predicted probability", fontsize=fontsize-1)
    # ax.set_ylabel("Fraction of positives", fontsize=fontsize-1)
    ax.set_xlabel("Probability", fontsize=fontsize-1)
    ax.set_ylabel("Accuracy", fontsize=fontsize-1)
    ax.set_ylim([0.0, 1.0])
    ax.set_xlim([0.0, 1.0]) # Set x-axis limits to 0-1
    ax.legend(loc="upper left", fontsize=fontsize)
    # ax.set_title("Reliability Diagram")
    ax.grid(True, linestyle='--', alpha=0.6)

    # If we created the figure, show it.
    if ax is None:
        plt.show()

def plot_multiple_reliability_diagrams(results, n_bins=15, main_title="Comparative Reliability Diagrams", fontsize=12):
    """
    Plots reliability diagrams for multiple models side-by-side.

    Args:
        results (dict): A dictionary where keys are model names and values are
                        tuples of (y_true, y_prob).
        n_bins (int): Number of bins for the calibration curve.
        main_title (str): The main title for the figure.
    """
    num_models = len(results)
    fig, axes = plt.subplots(1, num_models, figsize=(5 * num_models, 5), sharey=True)
    if num_models == 1:
        axes = [axes]

    colors = ['#1f77b4', '#d62728', '#2ca02c'] # Blue, Red, Green

    fig.suptitle(main_title, fontsize=16, y=1.03)

    for i, (ax, (model_name, (y_true, y_prob))) in enumerate(zip(axes, results.items())):
        # Convert to tensors if they are numpy arrays
        if isinstance(y_true, np.ndarray):
            y_true = torch.from_numpy(y_true)
        if isinstance(y_prob, np.ndarray):
            y_prob = torch.from_numpy(y_prob)

        # Use the existing single plot function
        plot_reliability_diagram(
            true_labels_tensor=y_true,
            all_outputs_tensor=y_prob,
            ax=ax,
            n_bins=n_bins,
            facecolor=colors[i % len(colors)],
            model_name=model_name,
            apply_sigmoid=False, # Probabilities are already computed
            fontsize=fontsize
        )
        # ax.set_title(model_name)
        if i > 0:
            ax.set_ylabel('') # Hide y-label for subsequent plots

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, ax,
                          normalize_colors=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          vmin=None, vmax=None,
                          show_yaxis=True,
                          fontsize=12):
    """
    Plots a single confusion matrix on a given matplotlib axis.
    Displays both count and percentage in each cell.
    """
    # Compute confusion matrices
    cm_counts = confusion_matrix(y_true, y_pred)
    # Calculate percentages - handle division by zero for rows with no samples
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_percent = cm_counts.astype('float') / cm_counts.sum(axis=1)[:, np.newaxis]
        cm_percent = np.nan_to_num(cm_percent)

    # Determine which matrix to use for coloring
    cm_for_display = cm_percent if normalize_colors else cm_counts

    im = ax.imshow(cm_for_display, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)

    # Set title and x-axis label
    ax.set_title(title, fontsize=fontsize+2)
    ax.set_xlabel('Predicted label', fontsize=fontsize)

    # Configure ticks
    ax.set(xticks=np.arange(cm_counts.shape[1]),
           yticks=np.arange(cm_counts.shape[0]),
           xticklabels=classes)
    
    # Conditionally show and rotate y-axis labels
    if show_yaxis:
        ax.set_ylabel('True label', fontsize=fontsize)
        ax.set_yticklabels(classes, rotation=90, va="center")
    else:
        ax.set_yticklabels([])


    # Ensure x-axis labels are horizontal
    ax.set_xticklabels(classes, rotation=0, ha="center")

    # Loop over data dimensions and create text annotations with both count and percentage
    # Determine the threshold for text color based on the normalized or absolute color scale
    thresh = (vmax - vmin) / 2. + vmin if vmax is not None else cm_for_display.max() / 2.
    for i in range(cm_counts.shape[0]):
        for j in range(cm_counts.shape[1]):
            count = cm_counts[i, j]
            percent = cm_percent[i, j]
            display_val = cm_for_display[i, j]
            ax.text(j, i, f"{count}\n({percent:.1%})",
                    ha="center", va="center",
                    # Set text color to white on dark cells, black on light cells
                    color="white" if display_val > thresh else "black", fontsize=fontsize)
    
    return im

def plot_multiple_confusion_matrices(results, class_names, normalize_colors=False, main_title="Model Comparison", cmap=plt.cm.Blues, fontsize=12):
    num_models = len(results)
    
    # Calculate global min/max for color normalization if requested
    global_vmin, global_vmax = None, None
    if normalize_colors:
        all_matrices = [
            confusion_matrix(y_true, (y_prob >= 0.5).astype(int)) 
            for model_name, (y_true, y_prob) in results.items()
        ]
        global_vmin = min(m.min() for m in all_matrices)
        global_vmax = max(m.max() for m in all_matrices)

    fig, axes = plt.subplots(1, num_models, figsize=(4 * num_models, 5))
    if num_models == 1:
        axes = [axes]
    
    fig.suptitle(main_title, fontsize=16, y=1.03)

    for i, (ax, (model_name, (y_true, y_prob))) in enumerate(zip(axes, results.items())):
        y_pred = (y_prob >= 0.5).astype(int)
        
        plot_confusion_matrix(
            y_true, y_pred, classes=class_names, ax=ax, 
            title=model_name, cmap=cmap, vmin=global_vmin, vmax=global_vmax,
            show_yaxis=(i==0), # Only show y-axis on the first plot
            fontsize=fontsize
        )
        # Set tick label font size
        ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_confusion_matrix_simple(y_true, y_pred, classes,
                          ax=None,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues,
                          fontsize=12):
    """
    Original simple version that plots a single confusion matrix with its own colorbar.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'
    if ax is None:
        fig, ax = plt.subplots()
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        with np.errstate(divide='ignore', invalid='ignore'):
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)
        cm_display = cm_normalized
    else:
        cm_display = cm
    im = ax.imshow(cm_display, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Set fontsize for tick labels (class labels)
    ax.tick_params(axis='both', which='major', labelsize=fontsize-2)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm_display.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm_display[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm_display[i, j] > thresh else "black", fontsize=fontsize)
    return ax

def plot_multiple_confusion_matrices_simple(results, class_names, normalize=False, main_title="Model Comparison", cmap=plt.cm.Blues):
    """
    Original simple version: Plots multiple confusion matrices, each with its own colorbar and axes.
    """
    num_models = len(results)
    fig, axes = plt.subplots(1, num_models, figsize=(7 * num_models, 6))
    if num_models == 1:
        axes = [axes]
    for ax, (model_name, (y_true, y_pred)) in zip(axes, results.items()):
        plot_confusion_matrix_simple(y_true, y_pred,
                                     classes=class_names,
                                     ax=ax,
                                     normalize=normalize,
                                     title=model_name,
                                     cmap=cmap)
    fig.suptitle(main_title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_beta_distribution(k=22.0, epsilon=0.8, alpha=None, beta_param=None, title=None):
    """
    Plot beta distribution with given parameters
    
    Args:
        k: base certainty parameter
        epsilon: epsilon_beta parameter
        alpha: direct alpha parameter (optional)
        beta_param: direct beta parameter (optional)
        title: plot title
    """
    
    # If alpha and beta are not provided, derive from k and epsilon
    if alpha is None or beta_param is None:
        # This is a guess based on your code structure - you may need to adjust
        # based on your actual parameter transformation
        alpha = k * epsilon
        beta_param = k * (1 - epsilon)
        derived_params = True
    else:
        derived_params = False
    
    # Ensure positive parameters
    alpha = max(alpha, 0.01)
    beta_param = max(beta_param, 0.01)
    
    # Create x values
    x = np.linspace(0.001, 0.999, 1000)
    
    # Create beta distribution
    beta_dist = beta(alpha, beta_param)
    
    # Calculate PDF
    pdf = beta_dist.pdf(x)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot PDF
    plt.subplot(2, 2, 1)
    plt.plot(x, pdf, 'b-', linewidth=2, label=f'PDF')
    plt.fill_between(x, pdf, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title(f'Beta Distribution PDF\nα={alpha:.3f}, β={beta_param:.3f}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Plot CDF
    plt.subplot(2, 2, 2)
    cdf = beta_dist.cdf(x)
    plt.plot(x, cdf, 'r-', linewidth=2, label='CDF')
    plt.xlabel('x')
    plt.ylabel('Cumulative Probability')
    plt.title('Beta Distribution CDF')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Statistics
    mean = beta_dist.mean()
    var = beta_dist.var()
    std = beta_dist.std()
    mode = (alpha - 1) / (alpha + beta_param - 2) if alpha > 1 and beta_param > 1 else "N/A"
    
    # Plot histogram of samples
    plt.subplot(2, 2, 3)
    samples = beta_dist.rvs(size=10000)
    plt.hist(samples, bins=50, density=True, alpha=0.7, color='green', label='Samples')
    plt.plot(x, pdf, 'b-', linewidth=2, label='True PDF')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.title('Samples vs True PDF')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Text summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    if derived_params:
        param_text = f"Input Parameters:\nk = {k}\nepsilon = {epsilon}\n\n"
    else:
        param_text = "Direct Parameters:\n\n"
    
    summary_text = f"""{param_text}Beta Distribution Parameters:
α (alpha) = {alpha:.4f}
β (beta) = {beta_param:.4f}

Statistics:
Mean = {mean:.4f}
Variance = {var:.4f}
Std Dev = {std:.4f}
Mode = {mode if isinstance(mode, str) else f'{mode:.4f}'}

Risk Score Mean = α/(α+β) = {mean:.4f}
Concentration = α+β = {alpha + beta_param:.4f}
"""
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    if title:
        plt.suptitle(title, fontsize=14, fontweight='bold')
    else:
        plt.suptitle(f'Beta Distribution Analysis (k={k}, ε={epsilon})', 
                     fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return alpha, beta_param, mean, var

def compare_beta_distributions(params_list, labels=None):
    """
    Compare multiple beta distributions
    
    Args:
        params_list: List of (k, epsilon) tuples or (alpha, beta) tuples
        labels: List of labels for each distribution
    """
    
    plt.figure(figsize=(15, 10))
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    x = np.linspace(0.001, 0.999, 1000)
    
    if labels is None:
        labels = [f'Distribution {i+1}' for i in range(len(params_list))]
    
    # Plot PDFs
    plt.subplot(2, 2, 1)
    for i, params in enumerate(params_list):
        if len(params) == 2:
            k, epsilon = params
            alpha = k * epsilon
            beta_param = k * (1 - epsilon)
            label = f'{labels[i]} (k={k}, ε={epsilon:.2f})'
        else:
            alpha, beta_param = params[:2]
            label = f'{labels[i]} (α={alpha:.2f}, β={beta_param:.2f})'
        
        alpha = max(alpha, 0.01)
        beta_param = max(beta_param, 0.01)
        
        beta_dist = beta(alpha, beta_param)
        pdf = beta_dist.pdf(x)
        
        plt.plot(x, pdf, color=colors[i % len(colors)], 
                linewidth=2, label=label)
    
    plt.xlabel('x')
    plt.ylabel('Probability Density')
    plt.title('Beta Distribution PDFs Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot CDFs
    plt.subplot(2, 2, 2)
    for i, params in enumerate(params_list):
        if len(params) == 2:
            k, epsilon = params
            alpha = k * epsilon
            beta_param = k * (1 - epsilon)
        else:
            alpha, beta_param = params[:2]
        
        alpha = max(alpha, 0.01)
        beta_param = max(beta_param, 0.01)
        
        beta_dist = beta(alpha, beta_param)
        cdf = beta_dist.cdf(x)
        
        plt.plot(x, cdf, color=colors[i % len(colors)], 
                linewidth=2, label=labels[i])
    
    plt.xlabel('x')
    plt.ylabel('Cumulative Probability')
    plt.title('Beta Distribution CDFs Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Statistics comparison
    plt.subplot(2, 1, 2)
    stats_data = []
    
    for i, params in enumerate(params_list):
        if len(params) == 2:
            k, epsilon = params
            alpha = k * epsilon
            beta_param = k * (1 - epsilon)
        else:
            alpha, beta_param = params[:2]
        
        alpha = max(alpha, 0.01)
        beta_param = max(beta_param, 0.01)
        
        beta_dist = beta(alpha, beta_param)
        mean = beta_dist.mean()
        std = beta_dist.std()
        
        stats_data.append([alpha, beta_param, mean, std, alpha + beta_param])
    
    # Create table
    plt.axis('off')
    table_data = []
    headers = ['Distribution', 'α', 'β', 'Mean', 'Std Dev', 'Concentration']
    
    for i, stats in enumerate(stats_data):
        row = [labels[i]] + [f'{val:.4f}' for val in stats]
        table_data.append(row)
    
    table = plt.table(cellText=table_data, colLabels=headers,
                     cellLoc='center', loc='center',
                     bbox=[0.1, 0.1, 0.8, 0.8])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    plt.suptitle('Beta Distributions Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()    

def plot_beta_uncertainty_analysis(beta_results, ax=None, main_title="Beta Model: Uncertainty Analysis", fontsize=11):
    """
    Visualizes the uncertainty information provided by the Beta model.
    Shows how the model expresses different levels of confidence in its predictions.
    
    Args:
        beta_results: Dictionary with keys 'labels', 'alpha_pred', 'beta_pred', 'risk_score'
    """
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Extract data
    y_true = beta_results['labels'].numpy()
    alpha = beta_results['alpha_pred'].numpy()
    beta_param = beta_results['beta_pred'].numpy()
    risk_score = beta_results['risk_score'].numpy()
    
    # Calculate uncertainty metrics
    # Variance of Beta distribution: (α*β) / ((α+β)²*(α+β+1))
    variance = (alpha * beta_param) / ((alpha + beta_param)**2 * (alpha + beta_param + 1))
    uncertainty = np.sqrt(variance)  # Standard deviation as uncertainty measure
    
    # Create scatter plot: risk score vs uncertainty, colored by true label
    colors = ['red' if label == 0 else 'blue' for label in y_true]
    labels = ['Negative (True)' if label == 0 else 'Positive (True)' for label in y_true]
    
    # Plot with transparency to handle overlapping points
    for true_label in [0, 1]:
        mask = y_true == true_label
        color = 'red' if true_label == 0 else 'blue'
        label = 'Negative Cases (True)' if true_label == 0 else 'Positive Cases (True)'
        
        ax.scatter(risk_score[mask], uncertainty[mask], 
                  c=color, alpha=0.6, s=20, label=label)
    
    ax.set_xlabel('Predicted Risk Score (Mean of Beta Distribution)', fontsize=fontsize)
    ax.set_ylabel('Prediction Uncertainty\n(Std Dev of Beta Distribution)', fontsize=fontsize)
    ax.set_title(main_title, fontsize=fontsize+2)
    ax.legend(fontsize=fontsize)
    ax.grid(True, alpha=0.3)
    
    # Add explanatory text
    ax.text(0.02, 0.98, 
            'Higher uncertainty = Less confident predictions\n' +
            'Beta model can express: "I predict 0.7 risk, but I\'m uncertain"\n' +
            'vs. baseline: "I predict 0.7 risk" (no uncertainty info)',
            transform=ax.transAxes, fontsize=fontsize, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    # Print summary statistics
    print(f"\n--- BETA MODEL UNCERTAINTY ANALYSIS ---")
    print(f"Risk score range: [{risk_score.min():.3f}, {risk_score.max():.3f}]")
    print(f"Uncertainty range: [{uncertainty.min():.3f}, {uncertainty.max():.3f}]")
    print(f"Mean uncertainty: {uncertainty.mean():.3f}")
    print(f"High uncertainty samples (>90th percentile): {np.sum(uncertainty > np.percentile(uncertainty, 90))}")
    print(f"Low uncertainty samples (<10th percentile): {np.sum(uncertainty < np.percentile(uncertainty, 10))}")
    
    return ax

def plot_beta_confidence_intervals(beta_results, ax, main_title="Beta Model: Prediction Confidence Intervals", n_samples=100, fontsize=11):
    """
    Shows confidence intervals for Beta model predictions.
    Demonstrates the rich uncertainty information that baseline models cannot provide.
    """
    # Extract data
    alpha = beta_results['alpha_pred'].numpy()
    beta_param = beta_results['beta_pred'].numpy()
    risk_score = beta_results['risk_score'].numpy()
    
    # Sample a subset for visualization
    indices = np.random.choice(len(alpha), min(n_samples, len(alpha)), replace=False)
    alpha_sample = alpha[indices]
    beta_sample = beta_param[indices]
    risk_sample = risk_score[indices]
    
    # Calculate confidence intervals using Beta distribution quantiles
    from scipy.stats import beta as beta_dist
    
    confidence_intervals = []
    for a, b in zip(alpha_sample, beta_sample):
        ci_lower = beta_dist.ppf(0.025, a, b)  # 2.5th percentile
        ci_upper = beta_dist.ppf(0.975, a, b)  # 97.5th percentile
        confidence_intervals.append((ci_lower, ci_upper))
    
    confidence_intervals = np.array(confidence_intervals)
    
    # Sort by risk score for better visualization
    sort_idx = np.argsort(risk_sample)
    risk_sorted = risk_sample[sort_idx]
    ci_sorted = confidence_intervals[sort_idx]
    
    # Plot
    x = np.arange(len(risk_sorted))
    ax.plot(x, risk_sorted, 'bo-', label='Predicted Risk (Mean)', markersize=4)
    ax.fill_between(x, ci_sorted[:, 0], ci_sorted[:, 1], 
                    alpha=0.3, color='blue', label='95% Confidence Interval')
    
    ax.set_xlabel('Sample Index (sorted by risk score)', fontsize=fontsize)
    ax.set_ylabel('Risk Probability', fontsize=fontsize)
    ax.set_title(main_title, fontsize=fontsize+2)
    ax.legend(fontsize=fontsize)
    ax.grid(True, alpha=0.3)
    
    # Add explanatory text
    ax.text(0.02, 0.98, 
            'Blue band = Uncertainty in each prediction\n' +
            'Wider bands = More uncertain predictions\n' +
            'Baseline models cannot provide this information',
            transform=ax.transAxes, fontsize=fontsize, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.8))
    
    return ax

def plot_prediction_interpretation(results, beta_results, ax, main_title="Prediction Interpretability", fontsize=12):
    """
    Compares the interpretability of predictions between models.
    """
    # Create example scenarios
    scenarios = [
        "High confidence, \nhigh risk",
        "High confidence, \nlow risk", 
        "Low confidence, \nmedium risk",
        "Very uncertain \nprediction"
    ]
    
    baseline_info = [
        "Risk: 0.85",
        "Risk: 0.15", 
        "Risk: 0.50",
        "Risk: 0.50"
    ]
    
    beta_info = [
        "Risk: 0.85 ± 0.05 (α=30, β=5)",
        "Risk: 0.15 ± 0.04 (α=5, β=30)",
        "Risk: 0.50 ± 0.15 (α=10, β=10)", 
        "Risk: 0.50 ± 0.25 (α=2, β=2)"
    ]
    
    # Clear the axis and create a text-based comparison
    ax.clear()
    ax.set_xlim(0, 10)
    ax.set_ylim(-1, len(scenarios) + 0.5)  # Extended y-limits for better spacing
    
    # Adjust spacing between rows
    row_spacing = 0.8  # Reduce spacing to prevent overlap
    
    for i, (scenario, baseline, beta) in enumerate(zip(scenarios, baseline_info, beta_info)):
        y = (len(scenarios) - 1 - i) * row_spacing
        
        # Scenario label
        ax.text(-0.3, y, scenario, fontweight='bold', fontsize=fontsize+2, va='center', ha='left')
        
        # Baseline model output - moved even further right to avoid overlap
        ax.text(4.2, y, baseline, fontsize=fontsize+2, va='center', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7))
        
        # Beta model output - adjusted positioning accordingly
        ax.text(8.3, y, beta, fontsize=fontsize+2, va='center', ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgreen', alpha=0.7))
    
    # Headers - positioned to match the text boxes
    header_y = len(scenarios) * row_spacing + 0.2
    ax.text(-0.3, header_y, 'Scenario', fontweight='bold', fontsize=fontsize+3, va='center', ha='left')
    ax.text(4.2, header_y, 'Baseline Models', fontweight='bold', fontsize=fontsize+3, va='center', ha='center')
    ax.text(8.3, header_y, 'Beta Model (Ours)', fontweight='bold', fontsize=fontsize+3, va='center', ha='center')
    
    ax.set_title(main_title, pad=10, fontsize=fontsize+2)
    ax.axis('off')
    
    return ax

def plot_meaningful_predictions_comparison(results, beta_results, main_title="Meaningful Predictions: Beta vs Baseline Models", fontsize=11):
    """
    Compares the meaningfulness of predictions between Beta model and baseline models.
    Shows how Beta model provides richer information.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(main_title, fontsize=16)
    
    # 1. Simple probability distributions (top left)
    ax1 = axes[0, 0]
    colors = ['#d62728', '#1f77b4', '#2ca02c'] # Red, Blue, Green
    for i, (model_name, (y_true, y_prob)) in enumerate(results.items()):
        color = colors[i % len(colors)]
        ax1.hist(y_prob, bins=50, range=(0, 1), alpha=0.6, 
               color=color, label=model_name, density=True)
    ax1.set_xlabel("Predicted Probability")
    ax1.set_ylabel("Density")
    ax1.set_title('Probability Distributions')
    ax1.legend(fontsize=fontsize)
    ax1.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. Beta uncertainty analysis (top right)  
    ax2 = axes[0, 1]
    plot_beta_uncertainty_analysis(beta_results, ax=ax2, main_title='Beta Model Uncertainty', fontsize=fontsize)
    
    # 3. Confidence intervals visualization (bottom left)
    ax3 = axes[1, 0]
    plot_beta_confidence_intervals(beta_results, ax=ax3, main_title='Beta Model: Confidence Intervals', fontsize=fontsize)
    
    # 4. Prediction interpretation (bottom right)
    ax4 = axes[1, 1]
    plot_prediction_interpretation(results, beta_results, ax=ax4, main_title='Prediction Interpretability', fontsize=fontsize)
    
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_qualitative_examples(images, beta_results, baseline_results, image_indices=None, image_titles=None,
                             main_title="Qualitative Examples: Beta Distribution Predictions",fontsize=12,ylim=80, img_row_height=4, dist_row_height=4):
    """
    Creates qualitative examples showing road scenes with their predicted Beta distributions.
    
    Args:
        images: Tensor or array of input images [N, C, H, W] or [N, H, W, C]
        beta_results: Dictionary with Beta model predictions
        baseline_results: Dictionary with baseline model predictions  
        image_indices: List of indices to show (if None, shows first 6)
        main_title: Title for the figure
    """
    if image_indices is None:
        image_indices = list(range(min(6, len(images))))
    
    n_examples = len(image_indices)
    fig, axes = plt.subplots(2, n_examples, figsize=(4*n_examples, img_row_height + dist_row_height), 
                             gridspec_kw={'height_ratios': [img_row_height, dist_row_height]})
    if n_examples == 1:
        axes = axes.reshape(2, 1)
    
    fig.suptitle(main_title, fontsize=16, y=0.95)
    
    # Extract data
    y_true = beta_results['labels'].numpy()
    alpha = beta_results['alpha_pred'].numpy()
    beta_param = beta_results['beta_pred'].numpy()
    beta_risk = beta_results['risk_score'].numpy()
    
    # Get baseline predictions (assuming it's one of the baseline models)
    baseline_risk = list(baseline_results.values())[0][1]  # Get probabilities from first baseline
    
    for i, idx in enumerate(image_indices):
        # Top row: Show the road scene image
        ax_img = axes[0, i]
        
        # Handle different image formats
        if isinstance(images, torch.Tensor):
            img = images[idx].cpu().numpy()
        else:
            img = images[idx]
            
        # Convert from [C, H, W] to [H, W, C] if needed
        if img.shape[0] == 3 and len(img.shape) == 3:
            img = np.transpose(img, (1, 2, 0))
        
        # Denormalize if needed (assuming ImageNet normalization)
        if img.max() <= 1.0 and img.min() >= -3.0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
        
        ax_img.imshow(img)
        if image_titles is None:
            ax_img.set_title(f'{"Crash" if y_true[idx] == 1 else "Safe"} (Ground Truth)', 
                            color='red' if y_true[idx] == 1 else 'green', fontweight='bold')
        else:
            ax_img.set_title(image_titles[idx], 
                             color='green' if (image_titles[idx] == "Easy Positive (TP)" or image_titles[idx] == "Easy Negative (TN)") else 'red', 
                             fontweight='bold')

        ax_img.axis('off')
        
        # Bottom row: Show Beta distribution and predictions
        ax_dist = axes[1, i]
        
        # Plot Beta distribution
        x = np.linspace(0, 1, 1000)
        from scipy.stats import beta as beta_dist
        
        alpha_val = alpha[idx]
        beta_val = beta_param[idx]
        beta_distribution = beta_dist(alpha_val, beta_val)
        pdf = beta_distribution.pdf(x)
        
        # Plot the distribution
        ax_dist.set_ylim(0, ylim)
        ax_dist.fill_between(x, pdf, alpha=0.3, color='blue')#, label='Beta Distribution')
        ax_dist.plot(x, pdf, 'b-', linewidth=2)
        
        # Mark the mean (risk score)
        risk_val = beta_risk[idx]
        ax_dist.axvline(risk_val, color='blue', linestyle='--', linewidth=2, 
                       label=f'Beta Risk: {risk_val:.3f}')
        
        # Mark baseline prediction
        baseline_val = baseline_risk[idx]
        ax_dist.axvline(baseline_val, color='red', linestyle=':', linewidth=2,
                       label=f'Baseline: {baseline_val:.3f}')
        
        # Add confidence interval
        ci_lower = beta_distribution.ppf(0.025)  # 2.5th percentile
        ci_upper = beta_distribution.ppf(0.975)  # 97.5th percentile
        ax_dist.axvspan(ci_lower, ci_upper, alpha=0.2, color='green', 
                       label=f'95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
        
        # Calculate uncertainty
        uncertainty = np.sqrt(beta_distribution.var())
        
        ax_dist.set_xlabel('Risk Probability')
        ax_dist.set_ylabel('Density')
        ax_dist.set_title(f'α={alpha_val:.2f}, β={beta_val:.2f}\nUncertainty: {uncertainty:.3f}')
        ax_dist.legend(fontsize=fontsize)
        ax_dist.grid(True, alpha=0.3)
        ax_dist.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.show()
    return fig

def plot_uncertainty_scenarios(beta_results, n_examples=8, main_title="Beta Model: Different Uncertainty Scenarios"):
    """
    Shows examples of different uncertainty scenarios from the Beta model.
    """
    # Extract data
    alpha = beta_results['alpha_pred'].numpy()
    beta_param = beta_results['beta_pred'].numpy()
    risk_score = beta_results['risk_score'].numpy()
    y_true = beta_results['labels'].numpy()
    
    # Calculate uncertainty for all samples
    from scipy.stats import beta as beta_dist
    uncertainties = []
    for a, b in zip(alpha, beta_param):
        uncertainties.append(np.sqrt(beta_dist(a, b).var()))
    uncertainties = np.array(uncertainties)
    
    # Select examples representing different scenarios
    scenarios = []
    
    # High confidence, high risk
    high_risk_confident = np.where((risk_score > 0.7) & (uncertainties < np.percentile(uncertainties, 25)))[0]
    if len(high_risk_confident) > 0:
        scenarios.append(('High Risk, High Confidence', high_risk_confident[0]))
    
    # High confidence, low risk  
    low_risk_confident = np.where((risk_score < 0.3) & (uncertainties < np.percentile(uncertainties, 25)))[0]
    if len(low_risk_confident) > 0:
        scenarios.append(('Low Risk, High Confidence', low_risk_confident[0]))
    
    # Medium risk, high uncertainty
    medium_risk_uncertain = np.where((risk_score > 0.4) & (risk_score < 0.6) & 
                                    (uncertainties > np.percentile(uncertainties, 75)))[0]
    if len(medium_risk_uncertain) > 0:
        scenarios.append(('Medium Risk, High Uncertainty', medium_risk_uncertain[0]))
    
    # High risk, high uncertainty (concerning case)
    high_risk_uncertain = np.where((risk_score > 0.6) & (uncertainties > np.percentile(uncertainties, 75)))[0]
    if len(high_risk_uncertain) > 0:
        scenarios.append(('High Risk, High Uncertainty', high_risk_uncertain[0]))
    
    # Fill remaining slots with diverse examples
    remaining_indices = np.random.choice(len(alpha), n_examples - len(scenarios), replace=False)
    for i, idx in enumerate(remaining_indices):
        scenarios.append((f'Example {len(scenarios) + 1}', idx))
    
    # Limit to requested number
    scenarios = scenarios[:n_examples]
    
    # Create the plot
    n_cols = 4
    n_rows = (len(scenarios) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(main_title, fontsize=16)
    
    x = np.linspace(0, 1, 1000)
    
    for i, (scenario_name, idx) in enumerate(scenarios):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Get parameters for this example
        alpha_val = alpha[idx]
        beta_val = beta_param[idx]
        risk_val = risk_score[idx]
        true_label = y_true[idx]
        uncertainty = uncertainties[idx]
        
        # Create Beta distribution
        beta_distribution = beta_dist(alpha_val, beta_val)
        pdf = beta_distribution.pdf(x)
        
        # Plot distribution
        color = 'red' if risk_val > 0.5 else 'green'
        ax.fill_between(x, pdf, alpha=0.3, color=color)
        ax.plot(x, pdf, color=color, linewidth=2)
        
        # Mark mean and confidence interval
        ax.axvline(risk_val, color=color, linestyle='--', linewidth=2)
        
        ci_lower = beta_distribution.ppf(0.025)
        ci_upper = beta_distribution.ppf(0.975)
        ax.axvspan(ci_lower, ci_upper, alpha=0.2, color='blue')
        
        # Formatting
        ax.set_title(f'{scenario_name}\nRisk: {risk_val:.3f} ± {uncertainty:.3f}\n' + 
                    f'True: {"Crash" if true_label == 1 else "Safe"}', fontsize=10)
        ax.set_xlabel('Risk Probability')
        ax.set_ylabel('Density')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
    
    # Hide empty subplots
    for i in range(len(scenarios), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()
    return fig

def plot_prediction_comparison_table(beta_results, baseline_results, n_examples=10, 
                                   main_title="Prediction Comparison: Beta vs Baseline"):
    """
    Creates a table comparing Beta model predictions with baseline predictions.
    """
    # Extract data
    y_true = beta_results['labels'].numpy()
    alpha = beta_results['alpha_pred'].numpy()
    beta_param = beta_results['beta_pred'].numpy()
    beta_risk = beta_results['risk_score'].numpy()
    
    # Get baseline predictions
    baseline_name = list(baseline_results.keys())[0]
    baseline_risk = baseline_results[baseline_name][1]
    
    # Calculate uncertainties
    from scipy.stats import beta as beta_dist
    uncertainties = []
    ci_widths = []
    for a, b in zip(alpha, beta_param):
        dist = beta_dist(a, b)
        uncertainties.append(np.sqrt(dist.var()))
        ci_lower = dist.ppf(0.025)
        ci_upper = dist.ppf(0.975)
        ci_widths.append(ci_upper - ci_lower)
    
    uncertainties = np.array(uncertainties)
    ci_widths = np.array(ci_widths)
    
    # Select diverse examples
    indices = np.random.choice(len(y_true), n_examples, replace=False)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('off')
    
    # Prepare table data
    headers = ['Sample', 'True Label', 'Baseline Risk', 'Beta Risk', 'Beta Uncertainty', 
               '95% CI Width', 'Beta Distribution', 'Interpretation']
    
    table_data = []
    for i, idx in enumerate(indices):
        true_label = 'Crash' if y_true[idx] == 1 else 'Safe'
        baseline_pred = f'{baseline_risk[idx]:.3f}'
        beta_pred = f'{beta_risk[idx]:.3f}'
        uncertainty = f'{uncertainties[idx]:.3f}'
        ci_width = f'{ci_widths[idx]:.3f}'
        beta_params = f'α={alpha[idx]:.2f}, β={beta_param[idx]:.2f}'
        
        # Generate interpretation
        if uncertainties[idx] > np.percentile(uncertainties, 75):
            interpretation = 'High uncertainty - needs review'
        elif uncertainties[idx] < np.percentile(uncertainties, 25):
            if beta_risk[idx] > 0.7:
                interpretation = 'High risk, confident'
            elif beta_risk[idx] < 0.3:
                interpretation = 'Low risk, confident'
            else:
                interpretation = 'Medium risk, confident'
        else:
            interpretation = 'Moderate uncertainty'
        
        row = [f'{i+1}', true_label, baseline_pred, beta_pred, uncertainty, 
               ci_width, beta_params, interpretation]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers,
                    cellLoc='center', loc='center',
                    bbox=[0, 0, 1, 1])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Color code based on uncertainty
    for i, row_data in enumerate(table_data):
        uncertainty_val = float(row_data[4])
        if uncertainty_val > np.percentile(uncertainties, 75):
            color = '#ffcccc'  # Light red for high uncertainty
        elif uncertainty_val < np.percentile(uncertainties, 25):
            color = '#ccffcc'  # Light green for low uncertainty
        else:
            color = '#ffffcc'  # Light yellow for medium uncertainty
        
        for j in range(len(headers)):
            table[(i+1, j)].set_facecolor(color)
    
    # Header styling
    for j in range(len(headers)):
        table[(0, j)].set_facecolor('#ccccff')
        table[(0, j)].set_text_props(weight='bold')
    
    ax.set_title(main_title, fontsize=16, pad=20)
    
    # Add legend
    legend_text = ('Color coding: Green = Low uncertainty (confident), '
                  'Yellow = Medium uncertainty, Red = High uncertainty (needs review)')
    ax.text(0.5, -0.05, legend_text, transform=ax.transAxes, 
            ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    plt.show()
    return fig

def create_qualitative_figure_6(images, beta_results, baseline_results, image_indices=None):
    """
    Creates the complete Figure 6 for the paper with multiple qualitative analysis components.
    """
    print("Generating Figure 6: Qualitative Examples...")
    
    # Main qualitative examples with road scenes
    if images is not None:
        print("  - Road scenes with Beta distributions...")
        plot_qualitative_examples(images, beta_results, baseline_results, image_indices,
                                 main_title="Figure 6a: Road Scenes with Beta Distribution Predictions")
    
    # Uncertainty scenarios
    print("  - Different uncertainty scenarios...")
    plot_uncertainty_scenarios(beta_results, n_examples=8,
                              main_title="Figure 6b: Beta Model Uncertainty Scenarios")
    
    # Prediction comparison table
    print("  - Prediction comparison table...")
    plot_prediction_comparison_table(beta_results, baseline_results, n_examples=10,
                                   main_title="Figure 6c: Detailed Prediction Comparison")
    
    print("Figure 6 generation complete!")