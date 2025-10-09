import os
import sys
import random
import logging
from collections import deque 
from typing import Dict, List, Tuple, Deque, Any, Optional, Callable 

import numpy as np
import torch
import torch.nn as nn 
from torch.optim import Optimizer 


logger = logging.getLogger(__name__)
if not logger.hasHandlers(): 
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    )


def setup_logging(
    config: Dict[str, Any],
    is_rank_zero: bool,
    phase_name: Optional[str] = None
) -> logging.Logger:
    """
    Sets up experiment logging.
    Only the rank-0 process creates file+console handlers;
    others inherit whatever’s already configured (or stay silent).
    """
    # 1) Determine level
    level_name = str(config.get("log_level", "INFO")).upper()
    level = getattr(logging, level_name, logging.INFO)

    # 2) Grab root logger
    root = logging.getLogger()
    if is_rank_zero:
        # Remove all existing handlers
        for h in root.handlers[:]:
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass

        # Ensure save_path & run_name exist
        save_path = config.get("save_path")
        run_name  = config.get("run_name")
        if not save_path or not run_name:
            raise ValueError("config must include 'save_path' and 'run_name' for logging setup")

        # Create log directory
        log_dir = os.path.join(save_path, run_name, "logs")
        os.makedirs(log_dir, exist_ok=True)

        # Filename using phase_name or default
        pname = phase_name or config.get("experiment_phase", "run")
        logfile = os.path.join(log_dir, f"{pname}.log")

        # 3) Configure basicConfig on root with force=True
        logging.basicConfig(
            level=level,
            format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
            handlers=[
                logging.FileHandler(logfile, mode="a"),
                logging.StreamHandler(sys.stdout),
            ],
            force=True,  # override any prior config
        )
        root.info(f"Rank 0 logging initialized. Writing to {logfile}")
    else:
        # Non-rank0: no changes, just inherit
        pass

    # Return a module-specific logger
    return logging.getLogger(__name__)


def set_seed(seed: int, rank: int = 0): # Add rank for per-process seeding if needed
    actual_seed = seed + rank # Make seed different for different DDP processes if desired
    random.seed(actual_seed)
    np.random.seed(actual_seed)
    torch.manual_seed(actual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(actual_seed)
        torch.cuda.manual_seed_all(actual_seed) 
    # For stricter reproducibility (can impact performance):
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    logger.info(f"Seed set to {actual_seed} (base: {seed}, rank: {rank})")


@torch.no_grad()
def compute_accuracy(pred_probs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Thresholds probabilities at 0.5 for binary accuracy.
    Args:
        pred_probs: Tensor of predicted probabilities, shape [B] or [B, 1].
        labels: Tensor of true labels (0 or 1), shape [B] or [B, 1].
    """
    if pred_probs.numel() == 0 or labels.numel() == 0:
        logger.warning("compute_accuracy called with empty tensors.")
        return 0.0 

    # Ensure tensors are on CPU for numpy conversion and comparison
    pred_probs_cpu = pred_probs.detach().cpu()
    labels_cpu = labels.detach().cpu()

    if pred_probs_cpu.ndim > 1 and pred_probs_cpu.shape[1] == 1:
        pred_probs_cpu = pred_probs_cpu.squeeze(-1)
    if labels_cpu.ndim > 1 and labels_cpu.shape[1] == 1:
        labels_cpu = labels_cpu.squeeze(-1)
        
    pred_binary = (pred_probs_cpu > 0.5).float()
    correct = (pred_binary == labels_cpu.float()).float()
    accuracy = correct.mean().item()
    return accuracy


def compute_ece(pred_probs: torch.Tensor, labels: torch.Tensor, n_bins: int = 15) -> float:
    """
    Expected Calibration Error for binary probabilities.
    """
    if pred_probs.numel() == 0 or labels.numel() == 0:
        logger.warning("compute_ece called with empty tensors.")
        return 0.0

    pred_probs_np = pred_probs.detach().cpu().view(-1).numpy()
    labels_np = labels.detach().cpu().view(-1).numpy().astype(int)
    
    valid_mask = ~np.isnan(pred_probs_np) & ~np.isinf(pred_probs_np)
    pred_probs_np = pred_probs_np[valid_mask]
    labels_np = labels_np[valid_mask]

    if len(pred_probs_np) == 0: return 0.0

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_low, bin_high = bin_boundaries[i], bin_boundaries[i+1]
        in_bin_mask = (pred_probs_np > bin_low) & (pred_probs_np <= bin_high)
        
        if np.any(in_bin_mask):
            prop_in_bin = np.mean(in_bin_mask)
            accuracy_in_bin = np.mean(labels_np[in_bin_mask])
            confidence_in_bin = np.mean(pred_probs_np[in_bin_mask])
            ece += np.abs(accuracy_in_bin - confidence_in_bin) * prop_in_bin
            
    return float(ece)


def adjust_learning_rate_action(
    optimizer: Optimizer,
    action_idx: int, # Expected: 1 for increase LR, 2 for decrease LR (from paper)
    current_optimizer_lr: float, # The current LR from optimizer.param_groups[0]['lr']
    lr_adjust_factor: float = 1.5, # Factor to multiply/divide by
    min_lr_bound: float = 1e-7,
    max_lr_bound: float = 1e-3,
) -> float:
    """
    Adjusts learning rate in the optimizer based on action_idx.
    Action 1 (Increase LR), Action 2 (Decrease LR). Other actions result in no change.
    Returns the new learning rate that was set.
    """
    target_lr = current_optimizer_lr

    if action_idx == 1: # Increase LR (as per paper's Action 1 description)
        target_lr = min(current_optimizer_lr * lr_adjust_factor, max_lr_bound)
    elif action_idx == 2: # Decrease LR (as per paper's Action 2 description)
        target_lr = max(current_optimizer_lr / lr_adjust_factor, min_lr_bound)
    # Else (e.g., action_idx 10 "Maintain settings"), no LR change by this function

    if abs(target_lr - current_optimizer_lr) > 1e-9: # If LR actually changes
        for param_group in optimizer.param_groups:
            param_group['lr'] = target_lr
        logger.info(f"Controller Action {action_idx}: LR changed from {current_optimizer_lr:.2e} to {target_lr:.2e}")
        return target_lr
    
    return current_optimizer_lr # Return current if no change applied

def adjust_fusion_alpha_action(
    weighted_average_module: nn.Module,
    action_idx: int, 
    raw_alpha_adjust_step: float = 0.25,
    min_raw_alpha_logit: float = -5.0, 
    max_raw_alpha_logit: float = 5.0,  
) -> float:
    """
    Adjusts the raw alpha logit up or down:
    - action_idx == 5: shift toward IMAGE  → increase raw logit
    - action_idx == 7: shift toward TEXT   → decrease raw logit
    """
    # sanity check
    if not hasattr(weighted_average_module, '_raw_alpha') or \
       not isinstance(weighted_average_module._raw_alpha, nn.Parameter):
        logger.error("No '_raw_alpha' parameter on module.")
        return 0.0

    current = weighted_average_module._raw_alpha.item()
    if action_idx == 5:
        # ↑ raw ⇒ α moves closer to 1 (image)
        target = current + raw_alpha_adjust_step
    elif action_idx == 7:
        # ↓ raw ⇒ α moves closer to 0 (text)
        target = current - raw_alpha_adjust_step
    else:
        # no change
        return current

    # clamp to [min, max]
    target = max(min_raw_alpha_logit, min(max_raw_alpha_logit, target))

    # apply if it actually moved
    if abs(target - current) > 1e-9:
        weighted_average_module._raw_alpha.data.fill_(target)
        new_alpha = torch.sigmoid(torch.tensor(target)).item()
        logger.info(f"Controller Action {action_idx}: Fusion _raw_alpha set to {target:.3f} (alpha ~ {new_alpha:.3f})")
        return target

    return current


def estimate_concept_drift_signal(
    accuracy_history: Deque[float], 
    window_size: int = 30, 
    comparison_split_ratio: float = 0.5, 
    min_required_history_for_calc: Optional[int] = None,
    significant_drop_threshold: float = 0.05 
) -> float:
    if min_required_history_for_calc is None:
        min_required_history_for_calc = window_size

    if not isinstance(accuracy_history, deque): # Basic type check
        logger.warning("estimate_concept_drift_signal: accuracy_history is not a deque.")
        return 0.0

    if len(accuracy_history) < min_required_history_for_calc:
        return 0.0 

    # Ensure enough data for comparison based on window size
    if len(accuracy_history) < window_size:
        logger.debug(f"Not enough history ({len(accuracy_history)}) for window size ({window_size}) in drift estimation.")
        return 0.0

    current_window_data = list(accuracy_history)[-window_size:] # Get the most recent items
    
    split_point = int(window_size * comparison_split_ratio)
    if not (0 < split_point < window_size): # Ensure valid split points
        logger.debug(f"Invalid split point ({split_point}) for window size ({window_size}).")
        return 0.0 

    old_perf_data = current_window_data[:split_point]
    new_perf_data = current_window_data[split_point:]

    if not old_perf_data or not new_perf_data: # Should be caught by split_point check
        return 0.0

    avg_old_accuracy = np.mean(old_perf_data) if old_perf_data else 0.0
    avg_new_accuracy = np.mean(new_perf_data) if new_perf_data else 0.0

    accuracy_drop = avg_old_accuracy - avg_new_accuracy # Positive if performance dropped
    
    drift_signal = 0.0
    if accuracy_drop > 0 and significant_drop_threshold > 1e-6: # Avoid division by zero
        drift_signal = accuracy_drop / significant_drop_threshold
    elif accuracy_drop > 0: # Drop exists but threshold is zero, signal strong drift
        drift_signal = accuracy_drop * 100 # Arbitrary large multiplier
        
    logger.debug(f"DriftEst: Hist({len(accuracy_history)}), Win({len(current_window_data)}), OldAcc({avg_old_accuracy:.3f}), NewAcc({avg_new_accuracy:.3f}), Drop({accuracy_drop:.3f}), SigDropThresh({significant_drop_threshold:.3f}), DriftSignal({drift_signal:.3f})")
    return max(0.0, drift_signal)


def compute_modality_stability_scores(
    text_modality_recent_accuracy: float = 0.7, 
    image_modality_recent_accuracy: float = 0.7, 
    text_modality_confidence: float = 0.8, 
    image_modality_confidence: float = 0.8, 
    **kwargs 
) -> Tuple[float, float]:
    s_text = max(0, text_modality_recent_accuracy * text_modality_confidence) 
    s_image = max(0, image_modality_recent_accuracy * image_modality_confidence)

    total_stability_proxy = s_text + s_image
    if total_stability_proxy > 1e-6: 
        norm_s_text = s_text / total_stability_proxy
        norm_s_image = s_image / total_stability_proxy
    else: 
        norm_s_text = 0.5
        norm_s_image = 0.5
        
    logger.debug(f"Modality Stability: Text={norm_s_text:.3f}, Image={norm_s_image:.3f}")
    return norm_s_text, norm_s_image


def calculate_action_cost(action_idx: int, config: Optional[Dict[str,Any]] = None) -> float:
    """
    Calculates a representative cost for a given controller action index.
    Align this with the 10 actions described in your paper if possible.
    """
    # Default costs if not in config
    cost_map = {
        1: 0.05, 2: 0.05, # LR adjustments
        3: 0.02, 4: 0.02, # Memory management (placeholder)
        5: 0.02, 7: 0.02, # Fusion adjustments (5 for text bias, 7 for image bias)
        6: 0.03,          # Preprocessing defense (placeholder)
        8: 0.06,          # Adversarial training (placeholder - high cost)
        9: 0.01,          # Lowering output confidence (placeholder - low cost)
        10: 0.0,         # Maintain current settings
    }
    if config and config.get("action_costs"): # Allow overriding via config
        cost_map.update(config["action_costs"])

    cost = cost_map.get(action_idx, 0.1) # Default cost for unknown actions
    logger.debug(f"Cost for action {action_idx}: {cost}")
    return cost

# Add sys import for setup_logging to use sys.stdout
import sys