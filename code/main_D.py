from __future__ import annotations
import argparse
import json
import yaml
import os
import logging
import random
import sys
import shutil
from datetime import datetime
from typing import Dict, Any, Tuple, List, Optional, Deque, Callable
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.metrics import f1_score

from data_utils import (
    LabeledDataset,
    UnlabeledDataset,
    build_dataloader,
    no_drift_fn,
    image_drift_degradation,
    text_drift_semantic_shift
)
from model import (
    TextEncoder,
    ImageEncoder,
    Fusion,
    WeightedAverage,
    build_ddp
)
from utility import (
    setup_logging,
    set_seed,
    compute_accuracy,
    compute_ece,
    adjust_learning_rate_action,
    adjust_fusion_alpha_action,
    estimate_concept_drift_signal,
    compute_modality_stability_scores,
    calculate_action_cost
)

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] (%(name)s) %(message)s",
    )

             

_DEFAULTS: Dict[str, Any] = dict(
    # --- Core Run Info ---
    run_name=f"ctamd_exp_D_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}",  
    save_path="./output/ctamd_experiments_D_metrics_theory/",
    mode= "train_drift_experiment_single_csv",
    seed= 42,

    # --- Data Configuration ---
    data_root= "/blue/woodard/share/adversarialAI/datasets",
    dataset_folder_name= "M3A",
    csv_filename_combined= "combined_data.csv",
    split_by_column= "year",                          
    phase1_split_type= "value_cutoff",                    
    phase1_split_value= 2014,                        

    # ─── Default Drift Config ─────────────────────────────────────────────────
    image_drift_type= "degradation",                
    image_drift_args_jpeg_quality= 70,
    image_drift_args_noise_std= 0.02,
    image_drift_args_blur_radius= 1.0,

    text_drift_type= "semantic_shift",
    text_drift_args_keyword_map= {"original": "drifted", "example": "illustration"},
    text_drift_args_append_phrase= "DRIFT_SIGNAL_TEXT", 
    apply_drift_to_specific_class= None, 

    # ─── Gradual Drift ─────────────────────────────────
    drift_is_gradual= False,
    drift_gradual_steps= 200,

    phase_C_sub_phases= [],   

    # ─── Batch & Unlabeled Data ────────────────────────────────────────────────
    batch_size= 128,
    use_unlabeled_data= False,
    csv_filename_unlabeled_source= "unlabeled_data_source.csv",

    # ─── Model Config ─────────────────────────────────────────────────────────
    text_model_name= "openai/clip-vit-base-patch32",
    vit_model_name= "openai/clip-vit-base-patch32",
    fusion_type= "weighted_average",
    initial_raw_alpha= 0.0,

    # ─── Optimizer ─────────────────────────────────────────────────────────────
    optimizer_type= "AdamW",                      
    base_lr= 5e-4,
    weight_decay= 0.001,

    # ─── Experiment Loop Steps ────────────────────────────────────────────────
    steps_phase1_initial_train= 1000,
    steps_phase2_adaptation=     5000,

    log_every=             15,
    save_every_phase1=    100,
    save_every_phase2=    100,
    force_retrain_phase1= False,   

    # ─── Controller Parameters ────────────────────────────────────────────────
    controller_acc_history_len=     50,
    controller_drift_window=        75,
    controller_drift_sig_drop=     0.1,
    controller_drift_comp_ratio=   0.50,
    controller_rule_high_drift_signal_threshold=     0.75,
    controller_rule_moderate_drift_signal_threshold= 0.30,
    controller_lr_adjust_factor=    1.5,
    controller_alpha_adjust_step=   0.05,
    controller_min_lr_bound=      1e-7,
    controller_max_lr_bound=      1e-3,
    controller_ref_image_drift_type= "degradation",
    controller_ref_text_drift_type=  "semantic_shift",
    controller_enable_lr_adapt=    True,
    controller_enable_alpha_adapt= True,


    # ─── Controller Cost Configuration ─────────────────────────────
    # cost per action index (1–10)
    controller_action_costs={1: 0.1, 2: 0.1, 5: 0.05, 7: 0.05, 10: 0.0},

    # ─── Plotting ─────────────────────────────────────────────────────────────
    generate_plots=True,
)

def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LS-OGD Concept Drift Experiment D: Enhanced Metrics & Theory Connection")
    p.add_argument("--config", type=str, help="Path to a YAML/JSON configuration file to override defaults.")
    temp_defaults_for_argparse = json.loads(json.dumps(_DEFAULTS))
    for key, default_value in temp_defaults_for_argparse.items():
        if isinstance(default_value, bool):
            if default_value: p.add_argument(f"--no-{key.replace('_', '-')}", action="store_false", dest=key)
            else: p.add_argument(f"--{key.replace('_', '-')}", action="store_true", dest=key)
            p.set_defaults(**{key: default_value})
        elif isinstance(default_value, dict) or isinstance(default_value, list): # Handle dict and list
             p.add_argument(f"--{key.replace('_', '-')}", type=str, default=None, help=f"JSON string for {key}. Default: '{json.dumps(default_value)}'")
        elif default_value is None: p.add_argument(f"--{key.replace('_', '-')}", type=str, default=None)
        else: p.add_argument(f"--{key.replace('_', '-')}", type=type(default_value), default=None)
    return p

def _load_external_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        if path.endswith((".yaml", ".yml")): return yaml.safe_load(f)
        elif path.endswith(".json"): return json.load(f)
        else: raise ValueError(f"Unsupported config file format: {path}")

def _deep_update(target_dict: Dict, source_dict: Dict):
    for key, value in source_dict.items():
        if isinstance(value, dict) and key in target_dict and isinstance(target_dict[key], dict): _deep_update(target_dict[key], value)
        else: target_dict[key] = value

def _cast_value_from_str(value_str: Optional[str], expected_type: type, key_for_warning: str = ""):
    if value_str is None: return None
    if expected_type == bool: return str(value_str).lower() in ['true', '1', 'yes', 'y']
    if expected_type == dict:
        try: return json.loads(str(value_str))
        except json.JSONDecodeError: logger.warning(f"CastWarn: Dict value '{key_for_warning}' ('{value_str}') invalid."); return None
    if expected_type == list:
         try: return json.loads(str(value_str))
         except json.JSONDecodeError: logger.warning(f"CastWarn: List value '{key_for_warning}' ('{value_str}') invalid."); return None
    try: return expected_type(value_str)
    except (ValueError, TypeError): logger.warning(f"CastWarn: Cannot cast '{key_for_warning}' ('{value_str}') to {expected_type}."); return None

def _merge_config(cli_args: argparse.Namespace) -> Dict[str, Any]:
    config = json.loads(json.dumps(_DEFAULTS)) # Load script defaults
    if cli_args.config: # Override with external config file if provided
        try:
            _deep_update(config, _load_external_config(cli_args.config))
        except Exception as e:
            logger.error(f"Error loading config file {cli_args.config}: {e}")

    # Override with CLI arguments
    for key, default_val_from_script_defaults in _DEFAULTS.items(): 
        if hasattr(cli_args, key):
            cli_val = getattr(cli_args, key) 

            # Check if the current key is for a boolean type based on _DEFAULTS
            is_bool_type_in_script_defaults = isinstance(default_val_from_script_defaults, bool)

            # Condition to decide if we should use cli_val:
            # 1. If it's a boolean type, cli_val (True/False) is always the intended value from argparse.
            # 2. If it's not a boolean type, only use cli_val if it's not None.
            #    (If cli_val is None for non-bool, it means arg was not specified, so we keep existing config value).
            should_update_from_cli = False
            if is_bool_type_in_script_defaults:
                should_update_from_cli = True # Booleans from CLI are always definitive
            elif cli_val is not None:
                should_update_from_cli = True # Non-None CLI value for non-bool means user specified it

            if should_update_from_cli:
                current_config_val = config.get(key)
                # Proceed with type casting for non-bools if necessary, similar to original logic
                # This simplified block assumes cli_val is the value to consider.
                # Type casting for dict/list from string is important here.
                value_to_set = cli_val
                expected_type_from_script = type(default_val_from_script_defaults)
                
                if not is_bool_type_in_script_defaults and \
                   cli_val is not None and \
                   not isinstance(cli_val, expected_type_from_script) and \
                   expected_type_from_script is not type(None):
                    # Attempt to cast cli_val if its type doesn't match the expected type from _DEFAULTS
                    # and it's not a boolean (booleans are handled by argparse action)
                    # and expected type is not None.
                    casted_val = _cast_value_from_str(str(cli_val), expected_type_from_script, key)
                    
                    # Use casted_val if casting was successful or if user explicitly passed "none"/"null"
                    if casted_val is not None or (isinstance(cli_val, str) and cli_val.lower() in ['none', 'null']):
                        value_to_set = casted_val
                    else:
                        # Casting failed for a user-provided value that wasn't "none" or "null"
                        # Log a warning and potentially skip an update or use original cli_val if appropriate
                        logger.warning(
                            f"CLI Overwrite: Failed to cast CLI arg '{key}' ('{cli_val}') to {expected_type_from_script}. "
                            f"Using raw CLI value '{cli_val}' if different from current config, or keeping current: '{current_config_val}'."
                        )
                        # Decide if value_to_set should remain cli_val or if we should skip update
                        # For simplicity, if casting fails for non "none"/"null", we might log and keep cli_val
                        # or handle more gracefully. The original code was also a bit ambiguous here.
                        # Let's assume for now value_to_set remains cli_val (original uncasted) if casted_val is None from failure
                        # and cli_val wasn't "none"/"null". The _cast_value_from_str might return the original on failure or None.
                        if casted_val is None and not (isinstance(cli_val, str) and cli_val.lower() in ['none', 'null']):
                             pass # Potentially skip update or use raw cli_val if that's intended on cast fail
                        else:
                            value_to_set = casted_val


                if value_to_set != current_config_val: # Update only if different
                    config[key] = value_to_set
    
    # --- Type Conversions ---
    # This section should run AFTER all overrides to finalize types.
    int_keys = ["seed", "batch_size", "steps_phase1_initial_train", "steps_phase2_adaptation", "log_every", "save_every_phase1", "save_every_phase2", "controller_acc_history_len", "controller_drift_window", "image_drift_args_jpeg_quality", "drift_gradual_steps"]
    float_keys = ["base_lr", "initial_raw_alpha", "weight_decay", "phase1_split_value", "image_drift_args_noise_std", "image_drift_args_blur_radius", "controller_drift_sig_drop", "controller_drift_comp_ratio", "controller_rule_high_drift_signal_threshold", "controller_rule_moderate_drift_signal_threshold", "controller_lr_adjust_factor", "controller_alpha_adjust_step", "controller_min_lr_bound", "controller_max_lr_bound"]
    bool_keys = ["use_unlabeled_data", "controller_enable_lr_adapt", "controller_enable_alpha_adapt", "generate_plots", "drift_is_gradual"]
    list_keys = ["phase_C_sub_phases"]
    dict_keys = ["controller_action_costs", "text_drift_args_keyword_map"] 

    for k in int_keys:
        if k in config and config[k] is not None:
            try:
                config[k] = int(float(config[k]))
            except (ValueError, TypeError) as e:
                logger.error(f"Config Error: Cannot cast '{k}' value '{config[k]}' to int. Keeping as is or using default if possible. Error: {e}")
                # Potentially set to default from _DEFAULTS if conversion fails for critical key like seed
                if k == "seed" and _DEFAULTS.get(k) is not None:
                    logger.warning(f"Falling back to default seed: {_DEFAULTS[k]}")
                    config[k] = _DEFAULTS[k] # Fallback for seed
        elif k == "seed" and config.get(k) is None: # Ensure seed specifically is not None
             logger.warning(f"Config Error: '{k}' is None. Setting to default: {_DEFAULTS.get(k, 42)}")
             config[k] = _DEFAULTS.get(k, 42)


    for k in float_keys:
        if k in config and config[k] is not None:
            try:
                config[k] = float(config[k])
            except (ValueError, TypeError) as e:
                logger.error(f"Config Error: Cannot cast '{k}' value '{config[k]}' to float. Error: {e}")
    for k in bool_keys:
        if k in config and not isinstance(config[k], bool): 
            config[k] = str(config[k]).lower() in ['true', '1', 'yes', 'y']
    for k in list_keys:
        if k in config and config[k] is not None and not isinstance(config[k], list):
            try:
                loaded_val = json.loads(str(config[k]))
                if isinstance(loaded_val, list):
                    config[k] = loaded_val
                else:
                    logger.error(f"Config Error: Value '{config[k]}' for list key '{k}' did not decode to a list. Keeping as is or using default.")
                    if isinstance(_DEFAULTS.get(k), list): config[k] = _DEFAULTS.get(k) # Fallback
            except json.JSONDecodeError:
                logger.error(f"Config Error: Cannot cast '{k}' value '{config[k]}' to list. Keeping as is or using default.")
                if isinstance(_DEFAULTS.get(k), list): config[k] = _DEFAULTS.get(k) # Fallback
    for k in dict_keys:
        if k in config and config[k] is not None and not isinstance(config[k], dict):
            try:
                loaded_val = json.loads(str(config[k]))
                if isinstance(loaded_val, dict):
                    config[k] = loaded_val
                else:
                    logger.error(f"Config Error: Value '{config[k]}' for dict key '{k}' did not decode to a dict. Keeping as is or using default.")
                    if isinstance(_DEFAULTS.get(k), dict): config[k] = _DEFAULTS.get(k) # Fallback
            except json.JSONDecodeError:
                logger.error(f"Config Error: Cannot cast '{k}' value '{config[k]}' to dict. Keeping as is or using default.")
                if isinstance(_DEFAULTS.get(k), dict): config[k] = _DEFAULTS.get(k) # Fallback

    if "apply_drift_to_specific_class" in config:
        val = config["apply_drift_to_specific_class"]
        if isinstance(val, str) and (val.lower() == "none" or val == ""):
            config["apply_drift_to_specific_class"] = None
        elif val is not None:
            try:
                config["apply_drift_to_specific_class"] = int(val)
            except (ValueError, TypeError):
                logger.error(f"Config Error: apply_drift_to_specific_class '{val}' invalid. Setting to None.")
                config["apply_drift_to_specific_class"] = None
        # If val is already None, it remains None.
    return config

# DDP Initialization
def init_ddp_environment() -> bool:
    if "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        try:
            if os.environ.get('MASTER_ADDR') is None: os.environ['MASTER_ADDR'] = 'localhost'
            if os.environ.get('MASTER_PORT') is None: os.environ['MASTER_PORT'] = '29500' 
            dist.init_process_group(backend=backend, init_method='env://')
            if torch.cuda.is_available():
                local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank()))
                torch.cuda.set_device(local_rank)
                logger.info(f"DDP Init: Rank {dist.get_rank()}/{dist.get_world_size()} GPU {local_rank} {backend}")
            else:
                logger.info(f"DDP Init (CPU): Rank {dist.get_rank()}/{dist.get_world_size()} {backend}")
            return True
        except Exception as e:
            logger.error(f"DDP Init Failed: {e}. Non-DDP mode.")
            # Clean up environment variables to prevent issues if retrying in non-DDP
            for v in ["WORLD_SIZE", "RANK", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"]:
                os.environ.pop(v, None)
            return False
    return False

def is_rank0() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0

# Checkpoint Utility 
def save_checkpoint(models: Dict[str, nn.Module], optimizer: optim.Optimizer, epoch_or_step: int | str, save_dir: str, config_to_save: Optional[Dict] = None, is_static_baseline: bool = False, metrics: Optional[Dict] = None):
    if not is_rank0(): return
    os.makedirs(save_dir, exist_ok=True)
    model_states = {name: (model.module if hasattr(model, 'module') else model).state_dict() for name, model in models.items()}
    ckpt_data = {'epoch_or_step': epoch_or_step, 'models_state_dict': model_states, 'optimizer_state_dict': optimizer.state_dict()}
    
    if config_to_save:
        ckpt_data['config'] = config_to_save
        if metrics:
            ckpt_data['metrics'] = metrics
            
    if str(epoch_or_step) == "final_phase1" and not is_static_baseline: ckpt_filename = "ctamd_adaptive_models_checkpoint_final_phase1.pt"
    elif is_static_baseline and str(epoch_or_step) == "final_phase1": ckpt_filename = "static_baseline_models_checkpoint_final_phase1.pt"
    else: prefix = "static_baseline" if is_static_baseline else "ctamd_adaptive"; ckpt_filename = f"{prefix}_models_checkpoint_{epoch_or_step}.pt"
    save_path_full = os.path.join(save_dir, ckpt_filename); torch.save(ckpt_data, save_path_full); logger.info(f"Saved checkpoint to {save_path_full}")

def load_checkpoint(models: Dict[str, nn.Module], optimizer: Optional[optim.Optimizer], ckpt_path: str, device: torch.device, strict_load: bool = True) -> int:
    if not os.path.exists(ckpt_path): logger.warning(f"Checkpoint not found: {ckpt_path}. Starting fresh."); return 0
    try: checkpoint = torch.load(ckpt_path, map_location=device); logger.info(f"Loading checkpoint from {ckpt_path}")
    except Exception as e: logger.error(f"Error loading checkpoint {ckpt_path}: {e}. Starting fresh."); return 0
    for name, model_instance in models.items():
        model_to_load = model_instance.module if hasattr(model_instance, 'module') else model_instance
        if name in checkpoint['models_state_dict']:
            try: model_to_load.load_state_dict(checkpoint['models_state_dict'][name], strict=strict_load); logger.info(f"Loaded weights for '{name}' {'(strict)' if strict_load else '(non-strict)'}.")
            except RuntimeError as e: logger.error(f"RuntimeError loading state_dict for '{name}': {e}. Strict: {strict_load}.")
        else: logger.warning(f"No weights found for model '{name}' in checkpoint.")
    if optimizer and 'optimizer_state_dict' in checkpoint:
        try: optimizer.load_state_dict(checkpoint['optimizer_state_dict']); logger.info("Loaded optimizer state.")
        except Exception as e: logger.error(f"Error loading optimizer state: {e}.")
    elif optimizer: logger.warning("No optimizer state found in checkpoint.")
    loaded_step = checkpoint.get('epoch_or_step', 0)
    if isinstance(loaded_step, str):
        try: numeric_part = loaded_step.split('_')[-1]; loaded_step = int(numeric_part) if numeric_part.isdigit() else 0
        except ValueError: logger.warning(f"Could not parse step from '{loaded_step}'. Defaulting to 0."); loaded_step = 0
    logger.info(f"Resuming/loaded from checkpoint step/epoch: {loaded_step}.")
    return loaded_step if isinstance(loaded_step, int) else 0

# Rule-Based Controller Logic
# Calculates and returns action cost
def get_rule_based_controller_action_and_apply(
    current_accuracy: float, accuracy_history: Deque[float],
    optimizer: optim.Optimizer, fusion_module: Fusion, config: Dict[str, Any]
) -> Tuple[int, float, float, float, float]: # Returns (action_idx, lr, alpha_prob, drift_signal, action_cost)
    current_lr = optimizer.param_groups[0]['lr']
    weighted_avg_submodule = None
    if fusion_module.fusion_type == 'weighted_average' and hasattr(fusion_module, 'weighted_average_submodule'): weighted_avg_submodule = fusion_module.weighted_average_submodule
    current_raw_alpha = weighted_avg_submodule._raw_alpha.item() if weighted_avg_submodule and hasattr(weighted_avg_submodule, '_raw_alpha') else 0.0
    drift_signal_val = estimate_concept_drift_signal(accuracy_history, window_size=config["controller_drift_window"], comparison_split_ratio=config["controller_drift_comp_ratio"], min_required_history_for_calc=config.get("controller_min_hist_drift", config["controller_drift_window"] // 2), significant_drop_threshold=config["controller_drift_sig_drop"])
    lr_action_idx, alpha_action_idx = 10, 10
    if config.get("controller_enable_lr_adapt", True):
        if drift_signal_val > config["controller_rule_high_drift_signal_threshold"]: lr_action_idx = 1; logger.debug(f"RuleCtrlr (LR): High drift ({drift_signal_val:.3f}). Action: Inc LR.")
        elif drift_signal_val > config["controller_rule_moderate_drift_signal_threshold"]: lr_action_idx = 2; logger.debug(f"RuleCtrlr (LR): Mod drift ({drift_signal_val:.3f}). Action: Dec LR.")
    if config.get("controller_enable_alpha_adapt", True) and weighted_avg_submodule:
        if drift_signal_val > config["controller_rule_moderate_drift_signal_threshold"]:
            if config.get("controller_ref_image_drift_type", "none") != "none" and config.get("image_drift_type") == config["controller_ref_image_drift_type"]: alpha_action_idx = 5; logger.debug(f"RuleCtrlr (Alpha): Assumed img drift ({drift_signal_val:.3f}), action: -> text.")
            elif config.get("controller_ref_text_drift_type", "none") != "none" and config.get("text_drift_type") == config["controller_ref_text_drift_type"]: alpha_action_idx = 7; logger.debug(f"RuleCtrlr (Alpha): Assumed txt drift ({drift_signal_val:.3f}), action: -> image.")
    final_lr = current_lr
    if config.get("controller_enable_lr_adapt", True): final_lr = adjust_learning_rate_action(optimizer, lr_action_idx, current_lr, config["controller_lr_adjust_factor"], config["controller_min_lr_bound"], config["controller_max_lr_bound"])
    final_raw_alpha = current_raw_alpha
    if config.get("controller_enable_alpha_adapt", True) and weighted_avg_submodule and alpha_action_idx != 10: final_raw_alpha = adjust_fusion_alpha_action(weighted_avg_submodule, alpha_action_idx, config["controller_alpha_adjust_step"])
    logged_action_idx = 10;
    if lr_action_idx != 10 and alpha_action_idx != 10: logged_action_idx = lr_action_idx # Or define combo action index
    elif lr_action_idx != 10: logged_action_idx = lr_action_idx
    elif alpha_action_idx != 10: logged_action_idx = alpha_action_idx
    if not config.get("controller_enable_lr_adapt",True): logged_action_idx = alpha_action_idx if alpha_action_idx != 10 else 10
    if not config.get("controller_enable_alpha_adapt",True): logged_action_idx = lr_action_idx if lr_action_idx != 10 else 10
    if not config.get("controller_enable_lr_adapt",True) and not config.get("controller_enable_alpha_adapt",True): logged_action_idx = 10

    # Calculate Action Cost
    action_cost = calculate_action_cost(logged_action_idx, config.get("controller_action_costs"))
    # Note: Assumes calculate_action_cost exists in utility and can use the config dict if needed

    final_alpha_prob = torch.sigmoid(torch.tensor(final_raw_alpha)).item() if weighted_avg_submodule else 0.5
    return logged_action_idx, final_lr, final_alpha_prob, drift_signal_val, action_cost


# Core Experiment Phase Loop
# Calculate and log new metrics (F1, ECE, delta_error, controller_cost)
def run_phase_loop(
    phase_name: str, models: Dict[str, nn.Module], optimizer: optim.Optimizer, dataloader: DataLoader,
    loss_fn: nn.Module, device: torch.device, config: Dict[str, Any], global_start_step: int,
    num_steps_for_this_phase: int, metrics_logger_dict: Dict[str, List[Any]], is_adaptive_phase: bool = False,
    accuracy_history_for_controller: Optional[Deque[float]] = None, is_evaluation_only: bool = False,
    checkpoint_save_dir: Optional[str] = None, save_every_n_steps: Optional[int] = 0
):
    if is_rank0():
        for key in ["f1", "ece", "delta_error_signal", "controller_cost", "drift_signal", "controller_action", "error_signal"]:
            metrics_logger_dict.setdefault(key, [])

    logger.info(f"--- Starting {phase_name} (Duration: {num_steps_for_this_phase} steps, Global Step Start: {global_start_step}) ---")
    if num_steps_for_this_phase <= 0:
        logger.warning(f"Skipping {phase_name}.")
        return

    pbar_disabled = not is_rank0() or logger.getEffectiveLevel() > logging.INFO
    pbar = tqdm(range(num_steps_for_this_phase), desc=f"{phase_name} ({'Eval' if is_evaluation_only else 'Train/Adapt'})", disable=pbar_disabled)
    data_iterator = iter(dataloader)
    fusion_module_actual = models["fusion"].module if hasattr(models["fusion"], 'module') else models["fusion"]
    weighted_avg_submodule_for_logging = fusion_module_actual.weighted_average_submodule if getattr(fusion_module_actual, 'fusion_type', None) == 'weighted_average' and hasattr(fusion_module_actual, 'weighted_average_submodule') else None

    previous_error_signal = None

    for step_in_phase in range(num_steps_for_this_phase):
        current_global_step = global_start_step + step_in_phase
        for m in models:
            models[m].train() if not is_evaluation_only else models[m].eval()

        try:
            texts_batch, imgs_batch, labels_batch, item_indices = next(data_iterator)
        except StopIteration:
            data_iterator = iter(dataloader)
            texts_batch, imgs_batch, labels_batch, item_indices = next(data_iterator)
        labels_batch_dev = labels_batch.unsqueeze(1).float().to(device)

        with torch.set_grad_enabled(not is_evaluation_only):
            text_emb, text_prob = models["text"](texts_batch)
            img_emb, img_prob = models["image"](imgs_batch)
            fused_output = models["fusion"](text_emb, text_prob, img_emb, img_prob)
            current_loss = torch.tensor(0.0, device=device)
            if not is_evaluation_only:
                current_loss = loss_fn(fused_output, labels_batch_dev).mean()

        with torch.no_grad():
            fused_probs_detached = fused_output.detach()
            current_accuracy = compute_accuracy(fused_probs_detached, labels_batch_dev)

        if accuracy_history_for_controller is not None:
            accuracy_history_for_controller.append(current_accuracy)

        error_signal = 1.0 - current_accuracy
        delta_error_signal = error_signal - previous_error_signal if previous_error_signal is not None else 0.0
        previous_error_signal = error_signal

        fused_probs_cpu = fused_probs_detached.cpu()
        # Apply sigmoid if logits
        if fused_probs_cpu.min() < 0 or fused_probs_cpu.max() > 1:
            fused_probs_cpu = torch.sigmoid(fused_probs_cpu)

        labels_flat = labels_batch_dev.cpu().long().numpy().flatten()
        preds_flat = (fused_probs_cpu > 0.5).long().numpy().flatten()

        current_f1 = f1_score(labels_flat, preds_flat, average='binary', zero_division=1)
        current_ece = compute_ece(fused_probs_cpu, labels_batch_dev.cpu().long())

        # Controller action
        logged_act, lr_log, alpha_log, drift_sig_log, action_cost_log = 10, optimizer.param_groups[0]['lr'], 0.5, 0.0, 0.0
        if is_adaptive_phase and not is_evaluation_only:
            logged_act, lr_log, alpha_log, drift_sig_log, action_cost_log = get_rule_based_controller_action_and_apply(
                current_accuracy, accuracy_history_for_controller, optimizer, fusion_module_actual, config
            )

        if not is_evaluation_only:
            optimizer.zero_grad()
            current_loss.backward()
            optimizer.step()

        # Logging
        if is_rank0():
            metrics_logger_dict["step"].append(current_global_step)
            metrics_logger_dict["accuracy"].append(current_accuracy)
            metrics_logger_dict["f1"].append(current_f1)
            metrics_logger_dict["ece"].append(current_ece)
            metrics_logger_dict["loss"].append(current_loss.item())
            metrics_logger_dict["lr"].append(lr_log)
            metrics_logger_dict["fusion_alpha"].append(alpha_log)
            metrics_logger_dict["drift_signal"].append(drift_sig_log)
            metrics_logger_dict["controller_action"].append(logged_act)
            metrics_logger_dict["controller_cost"].append(action_cost_log)
            metrics_logger_dict["error_signal"].append(error_signal)
            metrics_logger_dict["delta_error_signal"].append(delta_error_signal)

            if (step_in_phase + 1) % config["log_every"] == 0:
                log_items = [
                    f"P:{phase_name}",
                    f"S:{step_in_phase+1}/{num_steps_for_this_phase}(G:{current_global_step})",
                    f"Loss:{current_loss.item():.3f}" if not is_evaluation_only else "Loss:N/A",
                    f"Acc:{current_accuracy:.3f}",
                    f"F1:{current_f1:.3f}",
                    f"ECE:{current_ece:.3f}",
                    f"LR:{lr_log:.1e}",
                    f"Alpha:{alpha_log:.2f}"
                ]
                if is_adaptive_phase:
                    log_items += [f"DriftS:{drift_sig_log:.2f}", f"CtrlAct:{logged_act}", f"Cost:{action_cost_log:.2f}"]
                log_str = " | ".join(log_items)
                pbar.set_postfix_str(log_str, refresh=False)
                logger.info(log_str)

        if not is_evaluation_only and save_every_n_steps and save_every_n_steps > 0 and checkpoint_save_dir and (step_in_phase + 1) % save_every_n_steps == 0:
            save_checkpoint(models, optimizer, f"{phase_name}_step_{step_in_phase+1}", checkpoint_save_dir, config)
        pbar.update(1)

    pbar.close()
    logger.info(f"--- Finished {phase_name} ---")



# Main Experiment Orchestration
def main_experiment_orchestrator(config: Dict[str, Any]):
    # Setup 
    is_ddp_active = init_ddp_environment()
    if is_ddp_active and torch.cuda.is_available(): device = torch.device("cuda")
    elif torch.cuda.is_available(): device = torch.device("cuda")
    else: device = torch.device("cpu")
    setup_logging(config, is_rank0(), phase_name="setup_D")
    current_rank = dist.get_rank() if is_ddp_active else 0
    set_seed(config["seed"] + current_rank, rank=current_rank)
    logger.info(f"Rank {current_rank}, Device {str(device)}, Seed {config['seed'] + current_rank}")
    if is_rank0(): logger.info(f"Effective Config for {config.get('run_name', 'main_D_run')}: \n{yaml.dump(config, indent=2)}")

    work_dir_for_this_run = os.path.join(config["save_path"], config["run_name"])
    phase1_master_checkpoint_base_dir = os.path.join(config["save_path"], "common_phase1_model_store")
    phase1_final_checkpoint_filename = "ctamd_adaptive_models_checkpoint_final_phase1.pt"
    phase1_master_ckpt_full_path = os.path.join(phase1_master_checkpoint_base_dir, phase1_final_checkpoint_filename)

    if is_rank0():
        os.makedirs(work_dir_for_this_run, exist_ok=True)
        os.makedirs(phase1_master_checkpoint_base_dir, exist_ok=True)
        with open(os.path.join(work_dir_for_this_run, "effective_config.yaml"), "w") as f_cfg: yaml.dump(config, f_cfg, indent=2)

    # Load Data & Split
    combined_csv_full_path = os.path.join(config["data_root"], config["dataset_folder_name"], config["csv_filename_combined"])
    if not os.path.exists(combined_csv_full_path): logger.error(f"CSV {combined_csv_full_path} not found. Exit."); return
    try: full_df = pd.read_csv(combined_csv_full_path)
    except Exception as e: logger.error(f"Error reading CSV {combined_csv_full_path}: {e}. Exit."); return
    logger.info(f"Loaded {combined_csv_full_path}, rows: {len(full_df)}")
    if config.get("split_by_column") in full_df.columns: logger.info(f"Sorting by: {config['split_by_column']}"); full_df = full_df.sort_values(by=config["split_by_column"]).reset_index(drop=True)
    elif config.get("split_by_column"): logger.warning(f"Split column '{config['split_by_column']}' not found.")
    split_type, split_value = config["phase1_split_type"], config["phase1_split_value"]; df_phase1, df_phase2_source = pd.DataFrame(), pd.DataFrame()
    try: 
        if split_type=="ratio": split_idx=int(len(full_df)*float(split_value)); df_phase1, df_phase2_source = full_df.iloc[:split_idx], full_df.iloc[split_idx:]
        elif split_type=="count": split_idx=int(split_value); df_phase1, df_phase2_source = full_df.iloc[:split_idx], full_df.iloc[split_idx:]
        elif split_type=="value_cutoff" and config.get("split_by_column") in full_df.columns: col, val_str = config["split_by_column"], split_value; dtype=full_df[col].dtype; val = float(val_str) if pd.api.types.is_numeric_dtype(dtype) else str(val_str); df_phase1, df_phase2_source = full_df[full_df[col] <= val], full_df[full_df[col] > val]
        else: raise ValueError("Invalid split config.")
    except Exception as e: logger.error(f"Data split error: {e}. Default 50/50."); split_idx=len(full_df)//2; df_phase1, df_phase2_source = full_df.iloc[:split_idx], full_df.iloc[split_idx:]
    logger.info(f"Split -> P1: {len(df_phase1)}, P2 src: {len(df_phase2_source)}.")
    if len(df_phase1) == 0 or len(df_phase2_source) == 0: logger.error("Empty data phase(s). Exit."); return

    # Get Model Dims 
    try:
        with torch.no_grad(): text_feat_dim = TextEncoder(config["text_model_name"], device="cpu").backbone.config.hidden_size; img_feat_dim = ImageEncoder(config["vit_model_name"], device="cpu").backbone.config.hidden_size
    except Exception as e: logger.error(f"Failed get model dims: {e}. Exit."); return

    # Setup Phase 1 (Run training ONCE or load master checkpoint)
    metrics_log_p1_train = {} # Initialize, may remain empty if loaded
    models_p1, optimizer_p1 = None, None # Define scope
    if not os.path.exists(phase1_master_ckpt_full_path) or config.get("force_retrain_phase1", False):
        logger.info(f">>> Phase 1: Training Master Model (saving to {phase1_master_ckpt_full_path})...")
        text_m_p1=TextEncoder(config["text_model_name"], device, config).to(device); img_m_p1=ImageEncoder(config["vit_model_name"], device, config).to(device); fusion_m_p1=Fusion(config, text_feat_dim=text_feat_dim, image_feat_dim=img_feat_dim).to(device)
        text_m_p1_ddp=build_ddp(text_m_p1, device); img_m_p1_ddp=build_ddp(img_m_p1, device); fusion_m_p1_ddp=build_ddp(fusion_m_p1, device)
        models_p1={"text": text_m_p1_ddp, "image": img_m_p1_ddp, "fusion": fusion_m_p1_ddp}
        params_p1=[p for m in models_p1.values() for p in m.parameters() if p.requires_grad]
        if config["optimizer_type"].lower()=="adamw": optimizer_p1=optim.AdamW(params_p1, lr=config["base_lr"], weight_decay=config.get("weight_decay",0.01))
        elif config["optimizer_type"].lower()=="adam": optimizer_p1=optim.Adam(params_p1, lr=config["base_lr"])
        else: optimizer_p1=optim.SGD(params_p1, lr=config["base_lr"], momentum=config.get("sgd_momentum",0.9))
        dataset_paths_cfg={"data_root": config["data_root"], "dataset_folder_name": config["dataset_folder_name"]}
        dataset_p1=LabeledDataset(config=dataset_paths_cfg, initial_df=df_phase1); dataloader_p1=build_dataloader(dataset_p1, config["batch_size"], shuffle=not is_ddp_active, drop_last=is_ddp_active)
        metrics_log_p1_train = {"step": [], "loss": [], "accuracy": [], "lr": [], "fusion_alpha": [], "f1": [], "ece": [], "delta_error_signal": [], "controller_cost": [], "drift_signal": [], "controller_action": [], "error_signal": []} # Init with all keys
        bce_loss_fn = nn.BCELoss(reduction="none")
        run_phase_loop(phase_name="Phase1_MasterTrain", models=models_p1, optimizer=optimizer_p1, dataloader=dataloader_p1, loss_fn=bce_loss_fn, device=device, config=config, global_start_step=0, num_steps_for_this_phase=config["steps_phase1_initial_train"], metrics_logger_dict=metrics_log_p1_train, is_adaptive_phase=False, checkpoint_save_dir=os.path.join(phase1_master_checkpoint_base_dir, "intermediate_p1_checkpoints"), save_every_n_steps=config.get("save_every_phase1", 0))
        if is_rank0(): save_checkpoint(models_p1, optimizer_p1, "final_phase1", phase1_master_checkpoint_base_dir, config_to_save=config, is_static_baseline=False); logger.info(f"Phase 1 Master training complete. Model saved: {phase1_master_ckpt_full_path}")
    else: logger.info(f"Found existing Master Phase 1 checkpoint: {phase1_master_ckpt_full_path}. Reusing it.")
    if is_ddp_active: dist.barrier()


    # Metrics Logging Setup for Phase 2
    # Initialize with all keys expected by plotting
    all_metrics_log = {
        "phase1_train_ctamd": metrics_log_p1_train, # Use logged data if P1 run, else empty
        "phase2_eval_static": {"step": [], "loss": [], "accuracy": [], "f1": [], "ece": [], "lr": [], "fusion_alpha": [], "delta_error_signal": [], "error_signal": [], "drift_signal": [], "controller_action": [], "controller_cost": []},
        "phase2_adapt_ctamd": {"step": [], "loss": [], "accuracy": [], "f1": [], "ece": [], "lr": [], "fusion_alpha": [], "delta_error_signal": [], "error_signal": [], "drift_signal": [], "controller_action": [], "controller_cost": []},
    }
    accuracy_history_for_controller = deque(maxlen=config["controller_acc_history_len"])
    bce_loss_fn = nn.BCELoss(reduction="none")
    dataset_paths_cfg = {"data_root": config["data_root"], "dataset_folder_name": config["dataset_folder_name"]}


    # === PHASE 2: Static Model Evaluation ===
    logger.info(">>> Phase 2: Static Baseline Evaluation...")
    static_text_m=TextEncoder(config["text_model_name"], device, config).to(device); static_image_m=ImageEncoder(config["vit_model_name"], device, config).to(device); static_fusion_m=Fusion(config, text_feat_dim=text_feat_dim, image_feat_dim=img_feat_dim).to(device)
    static_models={"text": static_text_m, "image": static_image_m, "fusion": static_fusion_m}; dummy_opt_static=optim.AdamW([p for m in static_models.values() for p in m.parameters()], lr=config["base_lr"])
    if os.path.exists(phase1_master_ckpt_full_path):
        load_checkpoint(static_models, dummy_opt_static, phase1_master_ckpt_full_path, device)
        # Determine drift config for static eval (use first sub-phase or default)
        sub_phases = config.get("phase_C_sub_phases", [])
        static_eval_drift_cfg = sub_phases[0].get("drift_config", {}) if sub_phases else config
        img_dfn, img_dargs = no_drift_fn, {}; txt_dfn, txt_dargs = no_drift_fn, {}
        if static_eval_drift_cfg.get("image_drift_type") == "degradation": img_dfn, img_dargs = image_drift_degradation, {k.replace("image_drift_args_",""):v for k,v in static_eval_drift_cfg.items() if k.startswith("image_drift_args_")}
        if static_eval_drift_cfg.get("text_drift_type") == "semantic_shift": txt_dfn, txt_dargs = text_drift_semantic_shift, {k.replace("text_drift_args_",""):v for k,v in static_eval_drift_cfg.items() if k.startswith("text_drift_args_")}
        dataset_p2_static = LabeledDataset(config=dataset_paths_cfg, initial_df=df_phase2_source, image_drift_fn=img_dfn, text_drift_fn=txt_dfn, image_drift_args=img_dargs, text_drift_args=txt_dargs, apply_drift_to_specific_class=static_eval_drift_cfg.get("apply_drift_to_specific_class"))
        dataloader_p2_static = build_dataloader(dataset_p2_static, config["batch_size"], shuffle=False, drop_last=False)
        eval_steps = len(dataloader_p2_static); logger.info(f"Static eval steps: {eval_steps}")
        if eval_steps > 0: run_phase_loop( phase_name="Phase2_Eval_Static", models=static_models, optimizer=dummy_opt_static, dataloader=dataloader_p2_static, loss_fn=bce_loss_fn, device=device, config=config, global_start_step=config["steps_phase1_initial_train"], num_steps_for_this_phase=eval_steps, metrics_logger_dict=all_metrics_log["phase2_eval_static"], is_evaluation_only=True, is_adaptive_phase=False )
        else: logger.warning("Phase 2 dataloader empty for static eval.")
    else: logger.error(f"Master Phase 1 ckpt {phase1_master_ckpt_full_path} missing. Skipping static eval.")
    logger.info("Phase 2 Static Baseline Evaluation complete.")
    if is_ddp_active: dist.barrier()


    # === PHASE 2: LS-OGD Adaptation (Potentially Multiple Sub-Phases) ===
    logger.info(">>> Phase 2: LS-OGD Adaptation...")
    ctamd_text_m=TextEncoder(config["text_model_name"], device, config).to(device); ctamd_image_m=ImageEncoder(config["vit_model_name"], device, config).to(device); ctamd_fusion_m=Fusion(config, text_feat_dim=text_feat_dim, image_feat_dim=img_feat_dim).to(device)
    ctamd_text_m_ddp=build_ddp(ctamd_text_m, device); ctamd_image_m_ddp=build_ddp(ctamd_image_m, device); ctamd_fusion_m_ddp=build_ddp(ctamd_fusion_m, device)
    models_adapt={"text": ctamd_text_m_ddp, "image": ctamd_image_m_ddp, "fusion": ctamd_fusion_m_ddp}
    params_adapt=[p for m in models_adapt.values() for p in m.parameters() if p.requires_grad]
    if config["optimizer_type"].lower()=="adamw": optimizer_adapt=optim.AdamW(params_adapt, lr=config["base_lr"], weight_decay=config.get("weight_decay",0.01))
    elif config["optimizer_type"].lower()=="adam": optimizer_adapt=optim.Adam(params_adapt, lr=config["base_lr"])
    else: optimizer_adapt=optim.SGD(params_adapt, lr=config["base_lr"], momentum=config.get("sgd_momentum",0.9))
    
    sub_phase_start_times = [] # For plotting transition lines

    if os.path.exists(phase1_master_ckpt_full_path):
        load_checkpoint(models_adapt, optimizer_adapt, phase1_master_ckpt_full_path, device)
        accuracy_history_for_controller.clear(); prime_acc = config.get("controller_prime_accuracy", 0.75)
        for _ in range(min(accuracy_history_for_controller.maxlen, config.get("controller_prime_history_len", 10))): accuracy_history_for_controller.append(prime_acc)

        sub_phases_config = config.get("phase_C_sub_phases", [])
        current_global_step_tracker = config["steps_phase1_initial_train"]
        sub_phase_start_times.append(current_global_step_tracker)

        if sub_phases_config: # Run multiple sub-phases
            logger.info(f"Running Phase 2 adaptation with {len(sub_phases_config)} defined sub-phases.")
            for i, sub_phase in enumerate(sub_phases_config):
                sub_name = sub_phase.get("sub_name", f"Phase2_Sub{i+1}"); sub_duration = sub_phase.get("duration_steps", 100); sub_drift_cfg = sub_phase.get("drift_config", {}); logger.info(f"Starting sub-phase '{sub_name}' for {sub_duration} steps.")
                img_dfn, img_dargs = no_drift_fn, {}; txt_dfn, txt_dargs = no_drift_fn, {}
                if sub_drift_cfg.get("image_drift_type") == "degradation": img_dfn, img_dargs = image_drift_degradation, {k.replace("image_drift_args_",""):v for k,v in sub_drift_cfg.items() if k.startswith("image_drift_args_")}
                if sub_drift_cfg.get("text_drift_type") == "semantic_shift": txt_dfn, txt_dargs = text_drift_semantic_shift, {k.replace("text_drift_args_",""):v for k,v in sub_drift_cfg.items() if k.startswith("text_drift_args_")}
                dataset_p2_sub = LabeledDataset(config=dataset_paths_cfg, initial_df=df_phase2_source, image_drift_fn=img_dfn, text_drift_fn=txt_dfn, image_drift_args=img_dargs, text_drift_args=txt_dargs, apply_drift_to_specific_class=sub_drift_cfg.get("apply_drift_to_specific_class"))
                dataloader_p2_sub = build_dataloader(dataset_p2_sub, config["batch_size"], shuffle=False, drop_last=False)
                if len(dataloader_p2_sub) == 0: logger.warning(f"Dataloader for '{sub_name}' empty. Skipping."); current_global_step_tracker += sub_duration; continue
                run_phase_loop( phase_name=sub_name, models=models_adapt, optimizer=optimizer_adapt, dataloader=dataloader_p2_sub, loss_fn=bce_loss_fn, device=device, config=config, global_start_step=current_global_step_tracker, num_steps_for_this_phase=sub_duration, metrics_logger_dict=all_metrics_log["phase2_adapt_ctamd"], is_adaptive_phase=True, accuracy_history_for_controller=accuracy_history_for_controller, checkpoint_save_dir=os.path.join(work_dir_for_this_run, f"checkpoints_{sub_name}"), save_every_n_steps=config.get("save_every_phase2", 0) )
                current_global_step_tracker += sub_duration; sub_phase_start_times.append(current_global_step_tracker)
        else: # Run single Phase 2 adaptation phase
            logger.info(f"Running Phase 2 adaptation as single phase for {config['steps_phase2_adaptation']} steps.")
            img_dfn, img_dargs = no_drift_fn, {}; txt_dfn, txt_dargs = no_drift_fn, {}
            if config.get("image_drift_type") == "degradation": img_dfn, img_dargs = image_drift_degradation, {k.replace("image_drift_args_",""):v for k,v in config.items() if k.startswith("image_drift_args_")}
            if config.get("text_drift_type") == "semantic_shift": txt_dfn, txt_dargs = text_drift_semantic_shift, {k.replace("text_drift_args_",""):v for k,v in config.items() if k.startswith("text_drift_args_")}
            dataset_p2_single = LabeledDataset(config=dataset_paths_cfg, initial_df=df_phase2_source, image_drift_fn=img_dfn, text_drift_fn=txt_dfn, image_drift_args=img_dargs, text_drift_args=txt_dargs, apply_drift_to_specific_class=config.get("apply_drift_to_specific_class"))
            dataloader_p2_single = build_dataloader(dataset_p2_single, config["batch_size"], shuffle=False, drop_last=False)
            if len(dataloader_p2_single) > 0: run_phase_loop( phase_name="Phase2_Adapt_CTAMD", models=models_adapt, optimizer=optimizer_adapt, dataloader=dataloader_p2_single, loss_fn=bce_loss_fn, device=device, config=config, global_start_step=current_global_step_tracker, num_steps_for_this_phase=config["steps_phase2_adaptation"], metrics_logger_dict=all_metrics_log["phase2_adapt_ctamd"], is_adaptive_phase=True, accuracy_history_for_controller=accuracy_history_for_controller, checkpoint_save_dir=os.path.join(work_dir_for_this_run, "phase2_ctamd_checkpoints"), save_every_n_steps=config.get("save_every_phase2", 0) )
            else: logger.warning("Phase 2 dataloader empty for single adaptation phase.")
    else: logger.error(f"Master Phase 1 ckpt {phase1_master_ckpt_full_path} missing. Skipping LS-OGD adaptation.")
    logger.info("Phase 2 LS-OGD Adaptation complete.")
    if is_rank0(): save_checkpoint(models_adapt, optimizer_adapt, "final_phase2_adapted", work_dir_for_this_run, config_to_save=config)


    # Plotting Results
    if is_rank0() and config.get("generate_plots", True):
        logger.info(f"Generating and saving plots for run '{config['run_name']}'...")
        metrics_df_static = pd.DataFrame(all_metrics_log["phase2_eval_static"])
        metrics_df_adapt = pd.DataFrame(all_metrics_log["phase2_adapt_ctamd"])
        metrics_df_p1 = pd.DataFrame(all_metrics_log["phase1_train_ctamd"]) 

        # Plot 1: Accuracy Over Time (with sub-phase lines)
        plt.figure(figsize=(16, 8));
        if not metrics_df_p1.empty: plt.plot(metrics_df_p1["step"], metrics_df_p1["accuracy"], label="Phase 1 Train (if run)", linestyle='-', color='cornflowerblue', alpha=0.7)
        if not metrics_df_static.empty: plt.plot(metrics_df_static["step"], metrics_df_static["accuracy"], label="Static Baseline (Phase 2 Drifted)", color='red', marker='x', linestyle='--', markersize=4, alpha=0.6)
        if not metrics_df_adapt.empty: plt.plot(metrics_df_adapt["step"], metrics_df_adapt["accuracy"], label="LS-OGD (Phase 2 Adaptation)", color='green', linestyle='-', alpha=0.8)
        if sub_phase_start_times:
            for k, start_step in enumerate(sub_phase_start_times): plt.axvline(x=start_step, color='dimgray', linestyle=':', linewidth=1.5, label=f"Phase2 Start" if k==0 else None)
        plt.xlabel("Global Steps"); plt.ylabel("Accuracy"); plt.title(f"Performance Comparison"); plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.6); plt.ylim(-0.05, 1.05); plt.savefig(os.path.join(work_dir_for_this_run, "plot_D_accuracy_comparison.png")); plt.close()

        # Plot 2: F1 Score Over Time
        plt.figure(figsize=(16, 8));
        if not metrics_df_static.empty: plt.plot(metrics_df_static["step"], metrics_df_static["f1"], label="Static Baseline F1", color='red', marker='x', linestyle='--', markersize=4, alpha=0.6)
        if not metrics_df_adapt.empty: plt.plot(metrics_df_adapt["step"], metrics_df_adapt["f1"], label="LS-OGD F1", color='green', linestyle='-', alpha=0.8)
        if sub_phase_start_times: [plt.axvline(x=st, color='dimgray', ls=':', lw=1.5) for st in sub_phase_start_times]
        plt.xlabel("Global Steps (Phase 2)"); plt.ylabel("F1 Score"); plt.title(f"F1 Score Comparison"); plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.6); plt.ylim(-0.05, 1.05); plt.savefig(os.path.join(work_dir_for_this_run, "plot_D_f1_score_comparison.png")); plt.close()

        # Plot 3: ECE Over Time
        plt.figure(figsize=(16, 8));
        if not metrics_df_static.empty: plt.plot(metrics_df_static["step"], metrics_df_static["ece"], label="Static Baseline ECE", color='red', marker='x', linestyle='--', markersize=4, alpha=0.6)
        if not metrics_df_adapt.empty: plt.plot(metrics_df_adapt["step"], metrics_df_adapt["ece"], label="LS-OGD ECE", color='green', linestyle='-', alpha=0.8)
        if sub_phase_start_times: [plt.axvline(x=st, color='dimgray', ls=':', lw=1.5) for st in sub_phase_start_times]
        plt.xlabel("Global Steps (Phase 2)"); plt.ylabel("Expected Calibration Error (ECE)"); plt.title(f"ECE Comparison"); plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.6); plt.ylim(bottom=-0.01); plt.savefig(os.path.join(work_dir_for_this_run, "plot_D_ece_comparison.png")); plt.close()

        # Plot 4: Adaptation Signals (Alpha & LR) 
        if not metrics_df_adapt.empty:
            fig, ax1 = plt.subplots(figsize=(16, 8)); steps=metrics_df_adapt["step"]; color1='tab:orange'; ax1.set_xlabel("Global Steps (Phase 2)"); ax1.set_ylabel("Fusion Alpha (Prob)", color=color1); ax1.plot(steps, metrics_df_adapt["fusion_alpha"], color=color1, label="Fusion Alpha", alpha=0.8); ax1.tick_params(axis='y', labelcolor=color1); ax1.grid(True, linestyle='--', axis='y', alpha=0.3); ax1.set_ylim(0,1); ax2=ax1.twinx(); color2='tab:blue'; ax2.set_ylabel("Learning Rate", color=color2); ax2.plot(steps, metrics_df_adapt["lr"], color=color2, linestyle=':', label="Learning Rate", alpha=0.8); ax2.tick_params(axis='y', labelcolor=color2); ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1e'));
            if sub_phase_start_times: [ax1.axvline(x=st, color='dimgray', ls=':', lw=1.5) for st in sub_phase_start_times]
            plt.title(f"Adaptation Signals"); lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels(); ax2.legend(lines1 + lines2, labels1 + labels2, loc='best'); fig.tight_layout(); plt.savefig(os.path.join(work_dir_for_this_run, "plot_D_adaptation_signals.png")); plt.close()

        # Plot 5: Error & Drift Signals
        if not metrics_df_adapt.empty:
            fig, ax1 = plt.subplots(figsize=(16, 8)); steps=metrics_df_adapt["step"]; color1='purple'; ax1.set_xlabel("Global Steps (Phase 2)"); ax1.set_ylabel("Error Signal (1 - Accuracy)", color=color1); ax1.plot(steps, metrics_df_adapt["error_signal"], color=color1, label="Error Signal (1-Acc)", alpha=0.8); ax1.tick_params(axis='y', labelcolor=color1); ax1.grid(True, linestyle='--', axis='y', alpha=0.3); ax1.set_ylim(-0.05, 1.05); ax2=ax1.twinx(); color2='brown'; ax2.set_ylabel("Estimated Drift Signal", color=color2); ax2.plot(steps, metrics_df_adapt["drift_signal"], color=color2, linestyle='--', label="Est. Drift Signal", alpha=0.8); ax2.tick_params(axis='y', labelcolor=color2)
            if sub_phase_start_times: [ax1.axvline(x=st, color='dimgray', ls=':', lw=1.5) for st in sub_phase_start_times]
            plt.title(f"Error & Drift Signals"); lines1, labels1 = ax1.get_legend_handles_labels(); lines2, labels2 = ax2.get_legend_handles_labels(); ax2.legend(lines1 + lines2, labels1 + labels2, loc='best'); fig.tight_layout(); plt.savefig(os.path.join(work_dir_for_this_run, "plot_D_error_vs_drift.png")); plt.close()

        # Plot 6: Delta Error Signal (Lyapunov Proxy)
        if not metrics_df_adapt.empty and "delta_error_signal" in metrics_df_adapt.columns:
            plt.figure(figsize=(16, 8)); steps=metrics_df_adapt["step"]; delta_error = metrics_df_adapt["delta_error_signal"]
            plt.plot(steps, delta_error, label=r"$\Delta e(t) = e(t) - e(t-1)$", color='teal', linestyle='-', alpha=0.7)
            plt.axhline(0, color='black', linewidth=0.5, linestyle='--') # Zero line
            if sub_phase_start_times: [plt.axvline(x=st, color='dimgray', ls=':', lw=1.5) for st in sub_phase_start_times]
            plt.xlabel("Global Steps (Phase 2)"); plt.ylabel("Change in Error Signal"); plt.title(f"Delta Error Signal (Lyapunov Proxy)"); plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.6); plt.savefig(os.path.join(work_dir_for_this_run, "plot_D_delta_error_signal.png")); plt.close()

        # Plot 7: Controller Cost (Cumulative)
        if not metrics_df_adapt.empty and "controller_cost" in metrics_df_adapt.columns:
            plt.figure(figsize=(16, 8)); steps=metrics_df_adapt["step"]
            cumulative_cost = metrics_df_adapt["controller_cost"].cumsum()
            plt.plot(steps, cumulative_cost, label="Cumulative Controller Cost", color='magenta', linestyle='-', alpha=0.8)
            if sub_phase_start_times: [plt.axvline(x=st, color='dimgray', ls=':', lw=1.5) for st in sub_phase_start_times]
            plt.xlabel("Global Steps (Phase 2)"); plt.ylabel("Cumulative Cost"); plt.title(f"Cumulative Controller Cost"); plt.legend(loc='best'); plt.grid(True, linestyle='--', alpha=0.6); plt.savefig(os.path.join(work_dir_for_this_run, "plot_D_cumulative_controller_cost.png")); plt.close()

        logger.info(f"Plots saved in {work_dir_for_this_run}")

    if is_ddp_active: dist.destroy_process_group()
    logger.info(f"===== Experiment '{config['run_name']}' Finished Successfully =====")


if __name__ == "__main__":
    cli_args = _arg_parser().parse_args()
    final_config = _merge_config(cli_args)

    main_experiment_orchestrator(final_config)