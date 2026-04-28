# -*- coding: utf-8 -*-
"""
Main entry point for WMH Prediction experiments using ImageFlowNet.

This script provides a unified interface to run different experiments via command line.
All experiments follow the same pipeline with cross-validation.

Usage:
    python main.py --exp 1  # BL
    python main.py --exp 2  # L1
    python main.py --exp 3  # ODEY-BL
    python main.py --exp 4  # ODEM-BL
    python main.py --exp 5  # ODEY-L1
    python main.py --exp 6  # ODEM-L1
    
Available Experiments:
    1: BL
    2: L1
    3: ODEY-BL
    4: ODEM-BL
    5: ODEY-L1
    6: ODEM-L1
    
"""

import os
import torch
torch.cuda.empty_cache()
import argparse

# ============================================================
# === MEMORY OPTIMIZATION SETTINGS ===
# ============================================================

# Set PyTorch memory allocator to reduce fragmentation
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Enable TF32 for faster computation on Ampere GPUs (if available)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set cudnn benchmark for faster convolutions (if input sizes are constant)
torch.backends.cudnn.benchmark = True

print("⚙️ Memory optimization settings applied:")
print(f"   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
print(f"   - TF32 enabled: {torch.backends.cuda.matmul.allow_tf32}")
print(f"   - cuDNN benchmark: {torch.backends.cudnn.benchmark}")

# Import experiments directly (same folder)
from flair_to_flair_matrix import (
    ExperimentBLAllPairs,
    ExperimentL1AllPairs,
    ExperimentODEYBL,
    ExperimentODEMBL,
    ExperimentODEYL1,
    ExperimentODEML1,
)

# Registry of available experiments
EXPERIMENTS = {
    1: {
        "name": "flair_to_flair_bl",
        "description": "BL: all possible time points with original ImageFlowNet loss",
        "use_wmh": True,
        "class": ExperimentBLAllPairs,
    },
    2: {
        "name": "flair_to_flair_l1",
        "description": "L1: all possible time points + original ImageFlowNet loss + L1SSIM",
        "use_wmh": True,
        "class": ExperimentL1AllPairs,
    },
    3: {
        "name": "flair_to_flair_odey_bl",
        "description": "ODEY-BL: BL with ODE time scaling in years",
        "use_wmh": True,
        "class": ExperimentODEYBL,
    },
    4: {
        "name": "flair_to_flair_odem_bl",
        "description": "ODEM-BL: BL with ODE time scaling in months",
        "use_wmh": True,
        "class": ExperimentODEMBL,
    },
    5: {
        "name": "flair_to_flair_odey_l1",
        "description": "ODEY-L1: L1 variant with ODE time scaling in years",
        "use_wmh": True,
        "class": ExperimentODEYL1,
    },
    6: {
        "name": "flair_to_flair_odem_l1",
        "description": "ODEM-L1: L1 variant with ODE time scaling in months",
        "use_wmh": True,
        "class": ExperimentODEML1,
    },
}

# ============================================================
# === GLOBAL CONFIGURATION ===
# ============================================================

CONFIG = {
    # Dataset
    "ROOT_DIR": "/app/dataset/LBC1936",
    # "ROOT_DIR": "/disk/febrian/Edinburgh_Data/LBC1936",
    "FOLD_CSV": "4fold_split.csv",  # ✅ Stratified 4-fold split
    "SCAN_NAME_STAGE2": "Scan1Wave2",
    "SWINUNETR_MODELS_DIR": "swinunetr_models",  # Directory for pretrained SwinUNETR models
    
    # Training - MEMORY OPTIMIZED SETTINGS
    "BATCH_SIZE": 24,  # ⬇️ Reduced from 16 to 4 to save memory
    "LEARNING_RATE": 1e-4,
    "NUM_EPOCHS": 50,
    "MAX_SLICES": 48,
    "MAX_PATIENTS_PER_FOLD": 10000,
    
    # Thresholds and coefficients
    "RECON_PSNR_THR": 40.0,
    "CONTRASTIVE_COEFF": 0.1,
    "SMOOTHNESS_COEFF": 0.0,
    "LATENT_COEFF": 0.0,
    "INVARIANCE_COEFF": 0.0,
    "NO_L2": False,
    "NOISE_MAX_INTENSITY": 0.1,
    "ODE_MAX_T": 9.0,
    "SEG_LOSS_WEIGHT": 1.0,  # Weight for segmentation loss (Experiment 3)
    
    # Cross-validation - 4 folds for stratified split
    "CV_FOLDS": [1, 2, 3, 4],  # ✅ Changed from 5 folds to 4 folds
    "VAL_OFFSET": 1,  # Validation fold offset from test fold

    "FOLDS_TO_RUN": [1, 2, 3, 4],  # ✅ Changed from 5 folds to 4 folds

    # Segmentation flag
    "RUN_STAGE2": True,
    
    # Device
    "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Set derived values
CONFIG["K_FOLDS"] = len(CONFIG["CV_FOLDS"])

print(f"Using device: {CONFIG['DEVICE']}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run WMH Prediction experiments with ImageFlowNet',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Experiments:
    1: BL
    2: L1
    3: ODEY-BL
    4: ODEM-BL
    5: ODEY-L1
    6: ODEM-L1

Examples:
    python main.py --exp 1    # Run BL
    python main.py --exp 2    # Run L1
    python main.py --exp 3    # Run ODEY-BL
    python main.py --exp 4    # Run ODEM-BL
    python main.py --exp 5    # Run ODEY-L1
    python main.py --exp 6    # Run ODEM-L1
        """
    )
    parser.add_argument(
        '--exp',
        type=int,
        required=True,
        choices=sorted(EXPERIMENTS.keys()),
        help='Experiment number to run'
    )
    return parser.parse_args()

def main():
    """
    Main entry point for running experiments.
    Loads the selected experiment and executes it.
    """
    # Parse command line arguments
    args = parse_args()
    experiment_number = args.exp
    
    # Validate experiment number
    if experiment_number not in EXPERIMENTS:
        raise ValueError(
            f"Invalid experiment number: {experiment_number}. "
            f"Available experiments: {list(EXPERIMENTS.keys())}"
        )
    
    # Get experiment configuration
    exp_config = EXPERIMENTS[experiment_number]
    experiment_class = exp_config["class"]
    
    # Print experiment information
    print("\n" + "="*70)
    print(f"?? EXPERIMENT {experiment_number}: {exp_config['description']}")
    print("="*70)
    print(f"Name:        {exp_config['name']}")
    print(f"Use WMH:     {exp_config['use_wmh']}")
    print(f"Description: {exp_config['description']}")
    print("="*70 + "\n")
    
    # Initialize and run experiment
    experiment = experiment_class(experiment_number, exp_config, CONFIG)
    experiment.run()
    
    print("\n" + "="*70)
    print("? EXPERIMENT COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
