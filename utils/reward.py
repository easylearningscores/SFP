import torch
import torch.nn.functional as F
from typing import Callable, List

# -----------------------------------------------------------------------------
# All reward functions should handle inputs in the [B, T, C, H, W] format.
# -----------------------------------------------------------------------------

def calculate_csi(prediction: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> List[float]:
    """
    Calculates the Critical Success Index (CSI) as a reward.
    CSI focuses on evaluating the prediction accuracy of extreme events.

    Args:
        prediction (torch.Tensor): The model's prediction output, with shape [B, T, C, H, W].
        target (torch.Tensor): The ground truth target, with shape [B, T, C, H, W].
        threshold (float): The threshold used to binarize continuous values, representing an "event occurrence".

    Returns:
        List[float]: A list containing the CSI score for each sample, with length B.
                     A higher CSI indicates a better reward.
    """
    # Binarize the inputs
    pred_binary = (prediction > threshold).float()
    target_binary = (target > threshold).float()

    # Determine the dimensions to compute separately along the B dimension
    # We want to sum over the T, C, H, W dimensions, keeping the B dimension
    dims_to_sum = (1, 2, 3, 4)

    # Calculate the elements of the confusion matrix
    # TP (True Positives), also known as "hits"
    hits = (pred_binary * target_binary).sum(dim=dims_to_sum)
    
    # FN (False Negatives), also known as "misses"
    misses = ((1 - pred_binary) * target_binary).sum(dim=dims_to_sum)
    
    # FP (False Alarms)
    false_alarms = (pred_binary * (1 - target_binary)).sum(dim=dims_to_sum)

    # Calculate CSI
    # CSI = TP / (TP + FN + FP)
    csi_scores = hits / (hits + misses + false_alarms + 1e-7) # Add epsilon to prevent division by zero

    return csi_scores.tolist()


def calculate_negative_tke_error(prediction: torch.Tensor, target: torch.Tensor) -> List[float]:
    """
    Calculates the negative Turbulent Kinetic Energy (TKE) spectral error as a reward.
    This reward aims to encourage the model to maintain physical consistency. The smaller the error, 
    the larger the reward (negative error).

    Args:
        prediction (torch.Tensor): The model's prediction, with shape [B, T, C, H, W].
        target (torch.Tensor): The ground truth target, with shape [B, T, C, H, W].

    Returns:
        List[float]: A list containing the negative TKE spectral error for each sample.
    """
    batch_size = prediction.size(0)
    rewards = []

    for i in range(batch_size):
        # Process each sample individually
        pred_sample = prediction[i] # Shape: [T, C, H, W]
        target_sample = target[i]   # Shape: [T, C, H, W]
        
        # Simplified TKE calculation: TKE usually requires velocity components u, v, w.
        # Here, we assume the input channels represent these components or use a proxy.
        # For simplicity, we calculate the difference in energy spectra.
        # 1. Apply Fourier transform to the spatial dimensions (H, W)
        pred_fft = torch.fft.fft2(pred_sample, dim=(-2, -1))
        target_fft = torch.fft.fft2(target_sample, dim=(-2, -1))
        
        # 2. Calculate the power spectrum (square of the magnitude)
        pred_power_spectrum = torch.abs(pred_fft)**2
        target_power_spectrum = torch.abs(target_fft)**2
        
        # 3. Calculate the L2 error between the power spectra
        # Average over the T and C dimensions
        error = F.mse_loss(pred_power_spectrum, target_power_spectrum)
        
        # The reward is the negative error
        rewards.append(-error.item())
        
    return rewards


def calculate_ssim(prediction: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> List[float]:
    """
    Calculates the Structural Similarity Index (SSIM) as a reward.
    SSIM measures the similarity in structure, luminance, and contrast of images.

    Args:
        prediction (torch.Tensor): The model's prediction, with shape [B, T, C, H, W].
        target (torch.Tensor): The ground truth target, with shape [B, T, C, H, W].
        data_range (float): The dynamic range of the data (max_val - min_val).

    Returns:
        List[float]: A list containing the SSIM score for each sample.
    """
    B, T, C, H, W = prediction.shape
    
    prediction_reshaped = prediction.view(B * T, C, H, W)
    target_reshaped = target.view(B * T, C, H, W)

    ssim_scores_flat = _ssim_torch(prediction_reshaped, target_reshaped, data_range)
    
    ssim_scores = ssim_scores_flat.view(B, T).mean(dim=1)
    
    return ssim_scores.tolist()

def _ssim_torch(X, Y, data_range):
    K1, K2, C1, C2 = 0.01, 0.03, (K1*data_range)**2, (K2*data_range)**2
    mu_x = F.avg_pool2d(X, 3, 1, 1)
    mu_y = F.avg_pool2d(Y, 3, 1, 1)
    
    sigma_x = F.avg_pool2d(X**2, 3, 1, 1) - mu_x**2
    sigma_y = F.avg_pool2d(Y**2, 3, 1, 1) - mu_y**2
    sigma_xy = F.avg_pool2d(X*Y, 3, 1, 1) - mu_x*mu_y
    
    l = (2*mu_x*mu_y + C1) / (mu_x**2 + mu_y**2 + C1)
    c = (2*sigma_xy + C2) / (sigma_x + sigma_y + C2)
    ssim_map = l * c
    return ssim_map.mean(dim=[1, 2, 3]) 


# --- Reward Function Getter ---
def get_reward_function(name: str) -> Callable[[torch.Tensor, torch.Tensor], List[float]]:
    """
    A factory function that returns the corresponding reward function based on its name.

    Args:
        name (str): The name of the reward function ('csi', 'tke', 'ssim').

    Returns:
        Callable: The corresponding reward calculation function.
    """
    if name.lower() == 'csi':
        return calculate_csi
    
    elif name.lower() == 'tke':
        return calculate_negative_tke_error
        
    elif name.lower() == 'ssim':
        return calculate_ssim
    
    else:
        raise ValueError(f"Unknown reward function name: '{name}'. Options: 'csi', 'tke', 'ssim'.")