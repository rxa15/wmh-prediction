"""
Inference script for SwinUNETR-based WMH segmentation from FLAIR images.

This script loads trained models from 4-fold cross-validation and performs
inference on test data to generate WMH predictions.

Usage:
    python infer_swinunetr_4fold.py --root_dir <path> --models_dir <path> --output_dir <path>
    
Example:
    python infer_swinunetr_4fold.py --root_dir /disk/febrian/Edinburgh_Data/LBC1936 \\
                                     --models_dir swinunetr_models \\
                                     --output_dir swinunetr_predictions \\
                                     --test_fold 1
"""

import argparse
import os
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

try:
    from main import CONFIG as _MAIN_CONFIG
except Exception:
    _MAIN_CONFIG = {}

from utils import (
    SwinUNetSegmentation,
    FLAIREvolutionDataset,
    EvolutionSegmentationDataset,
    load_patient_split_map,
    _normalize_patient_id,
    _dice_coeff,
)

DEFAULT_ROOT_DIR = _MAIN_CONFIG.get("ROOT_DIR")
DEFAULT_SPLIT_CSV = _MAIN_CONFIG.get("FOLD_CSV", "4fold_split.csv")
DEFAULT_SCAN_NAME = _MAIN_CONFIG.get("SCAN_NAME_STAGE2", "Scan1Wave2")
DEFAULT_MODELS_DIR = "swinunetr_models"
DEFAULT_OUTPUT_DIR = "swinunetr_predictions"
DEFAULT_MAX_SLICES = _MAIN_CONFIG.get("MAX_SLICES", 48)


def load_model(checkpoint_path, device):
    """Load a trained SwinUNETR model from checkpoint."""
    model = SwinUNetSegmentation().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    print(f"✓ Loaded model from {checkpoint_path}")
    return model


def predict_batch(model, batch, device):
    """Run inference on a batch of FLAIR images."""
    with torch.no_grad():
        flair = batch["flair"].to(device)
        logits = model(flair)
        probs = torch.sigmoid(logits)
        pred_masks = (probs > 0.5).float()
    return probs, pred_masks


def compute_metrics(pred_masks, gt_masks):
    """Compute segmentation metrics (Dice, IoU, etc.)."""
    metrics = {}
    
    # Dice coefficient
    dice_scores = []
    for pred, gt in zip(pred_masks, gt_masks):
        dice = float(_dice_coeff(pred, gt).item())
        dice_scores.append(dice)
    
    metrics['dice_mean'] = np.mean(dice_scores)
    metrics['dice_std'] = np.std(dice_scores)
    metrics['dice_scores'] = dice_scores
    
    # IoU (Intersection over Union)
    iou_scores = []
    for pred, gt in zip(pred_masks, gt_masks):
        pred_flat = pred.reshape(-1)
        gt_flat = gt.reshape(-1)
        intersection = (pred_flat * gt_flat).sum()
        union = pred_flat.sum() + gt_flat.sum() - intersection
        iou = float(intersection / (union + 1e-6))
        iou_scores.append(iou)
    
    metrics['iou_mean'] = np.mean(iou_scores)
    metrics['iou_std'] = np.std(iou_scores)
    
    return metrics


def save_predictions(pred_masks, batch, output_dir, fold_name):
    """Save predicted masks as NIfTI files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, pred_mask in enumerate(pred_masks):
        patient_id = batch['patient_id'][i]
        slice_idx = batch['slice_idx'][i].item() if hasattr(batch['slice_idx'][i], 'item') else batch['slice_idx'][i]
        
        # Convert to numpy
        pred_np = pred_mask.cpu().numpy().squeeze()
        
        # Save as NIfTI
        filename = f"{fold_name}_{patient_id}_slice{slice_idx:03d}_pred.nii.gz"
        filepath = os.path.join(output_dir, filename)
        
        # Create NIfTI image
        nii_img = nib.Nifti1Image(pred_np, affine=np.eye(4))
        nib.save(nii_img, filepath)


def visualize_predictions(flair, pred_mask, gt_mask, output_path, patient_id, slice_idx):
    """Create visualization comparing prediction with ground truth."""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # FLAIR image
    axes[0].imshow(flair.squeeze(), cmap='gray')
    axes[0].set_title(f'FLAIR - {patient_id}\nSlice {slice_idx}')
    axes[0].axis('off')
    
    # Ground truth
    axes[1].imshow(gt_mask.squeeze(), cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Ground Truth WMH')
    axes[1].axis('off')
    
    # Prediction
    axes[2].imshow(pred_mask.squeeze(), cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Predicted WMH')
    axes[2].axis('off')
    
    # Overlay
    axes[3].imshow(flair.squeeze(), cmap='gray')
    axes[3].imshow(pred_mask.squeeze(), cmap='hot', alpha=0.5, vmin=0, vmax=1)
    axes[3].set_title('Prediction Overlay')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def run_inference_single_fold(
    model,
    test_loader,
    device,
    output_dir,
    fold_name,
    save_nifti=True,
    save_visualizations=True,
    num_vis_samples=10
):
    """Run inference on a single fold's test set."""
    model.eval()
    
    all_dice_scores = []
    all_iou_scores = []
    
    pred_dir = os.path.join(output_dir, fold_name, "predictions")
    vis_dir = os.path.join(output_dir, fold_name, "visualizations")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    vis_count = 0
    
    print(f"\n🔍 Running inference for {fold_name}...")
    for batch_idx, batch in enumerate(tqdm(test_loader, desc=f"Inference [{fold_name}]")):
        # Get predictions
        probs, pred_masks = predict_batch(model, batch, device)
        
        # Get ground truth
        gt_masks = batch["mask"].to(device)
        
        # Compute metrics
        for i in range(len(pred_masks)):
            dice = float(_dice_coeff(pred_masks[i], gt_masks[i]).item())
            all_dice_scores.append(dice)
            
            # IoU
            pred_flat = pred_masks[i].reshape(-1)
            gt_flat = gt_masks[i].reshape(-1)
            intersection = (pred_flat * gt_flat).sum()
            union = pred_flat.sum() + gt_flat.sum() - intersection
            iou = float(intersection / (union + 1e-6))
            all_iou_scores.append(iou)
        
        # Save predictions (optional)
        if save_nifti:
            save_predictions(pred_masks, batch, pred_dir, fold_name)
        
        # Save visualizations (first N samples)
        if save_visualizations and vis_count < num_vis_samples:
            for i in range(min(len(pred_masks), num_vis_samples - vis_count)):
                patient_id = batch['patient_id'][i]
                slice_idx = batch['slice_idx'][i].item() if hasattr(batch['slice_idx'][i], 'item') else batch['slice_idx'][i]
                
                flair_np = batch['flair'][i].cpu().numpy()
                pred_np = pred_masks[i].cpu().numpy()
                gt_np = gt_masks[i].cpu().numpy()
                
                vis_path = os.path.join(vis_dir, f"{patient_id}_slice{slice_idx:03d}.png")
                visualize_predictions(flair_np, pred_np, gt_np, vis_path, patient_id, slice_idx)
                vis_count += 1
            
            if vis_count >= num_vis_samples:
                break
    
    # Compute overall metrics
    metrics = {
        'dice_mean': np.mean(all_dice_scores),
        'dice_std': np.std(all_dice_scores),
        'dice_median': np.median(all_dice_scores),
        'iou_mean': np.mean(all_iou_scores),
        'iou_std': np.std(all_iou_scores),
        'iou_median': np.median(all_iou_scores),
        'num_samples': len(all_dice_scores),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run inference using trained SwinUNETR models for WMH segmentation."
    )
    parser.add_argument(
        "--root_dir",
        required=DEFAULT_ROOT_DIR is None,
        default=DEFAULT_ROOT_DIR,
        help="Dataset root containing Scan*_FLAIR_brain and Scan*_WMH folders.",
    )
    parser.add_argument(
        "--split_csv",
        default=DEFAULT_SPLIT_CSV,
        help="CSV with columns patient_ID and split_id/fold.",
    )
    parser.add_argument(
        "--scan_name",
        default=DEFAULT_SCAN_NAME,
        help="Which timepoint to run inference on.",
    )
    parser.add_argument(
        "--models_dir",
        default=DEFAULT_MODELS_DIR,
        help="Directory containing trained model checkpoints.",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for predictions and visualizations.",
    )
    parser.add_argument(
        "--test_fold",
        type=int,
        default=None,
        help="Specific test fold to run inference on (1-4). If not specified, runs all folds.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference.",
    )
    parser.add_argument(
        "--max_slices",
        type=int,
        default=DEFAULT_MAX_SLICES,
        help="Max slices per patient.",
    )
    parser.add_argument(
        "--val_offset",
        type=int,
        default=1,
        help="Validation split offset (must match training).",
    )
    parser.add_argument(
        "--require_wmh_presence",
        action="store_true",
        default=True,
        help="Keep only slices with WMH>0 (default: True).",
    )
    parser.add_argument(
        "--allow_empty_wmh",
        action="store_true",
        default=False,
        help="If set, includes slices even when WMH==0.",
    )
    parser.add_argument(
        "--save_nifti",
        action="store_true",
        default=False,
        help="Save predictions as NIfTI files.",
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        default=True,
        help="Save visualization images.",
    )
    parser.add_argument(
        "--num_vis_samples",
        type=int,
        default=10,
        help="Number of visualization samples to save per fold.",
    )
    parser.add_argument("--gpu", type=int, default=None, help="CUDA device index.")
    
    args = parser.parse_args()
    
    # Setup device
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}" if args.gpu is not None else "cuda")
        print(f"🖥️  Using device: {device}")
    else:
        device = torch.device("cpu")
        print("🖥️  Using device: CPU")
    
    # Load patient split map
    patient_to_split = load_patient_split_map(args.split_csv)
    all_splits = sorted(set(patient_to_split.values()))
    
    if len(all_splits) != 4:
        raise ValueError(f"Expected exactly 4 splits in {args.split_csv}, got {all_splits}")
    
    # Build base dataset
    require_wmh_presence = args.require_wmh_presence and (not args.allow_empty_wmh)
    
    base_dataset = FLAIREvolutionDataset(
        root_dir=args.root_dir,
        max_slices_per_patient=args.max_slices,
        use_wmh=True,
        training_pairs=[(args.scan_name, args.scan_name, 0.0)],
        require_wmh_presence=require_wmh_presence,
        use_flair_output=False,
    )
    
    # Precompute indices per split
    split_to_indices = {s: [] for s in all_splits}
    for i, item in enumerate(base_dataset.index_map):
        pid = _normalize_patient_id(item.get("patient_id", ""))
        s = patient_to_split.get(pid, None)
        if s in split_to_indices:
            split_to_indices[s].append(i)
    
    # Determine which folds to process
    if args.test_fold is not None:
        folds_to_process = [args.test_fold]
    else:
        folds_to_process = all_splits
    
    # Results collection
    all_results = []
    
    # Process each fold
    for test_split in folds_to_process:
        # Reconstruct val_split (must match training)
        val_split = all_splits[(all_splits.index(test_split) + int(args.val_offset)) % len(all_splits)]
        
        # Load model checkpoint
        checkpoint_name = f"wmh_swinunetr_test{test_split}_val{val_split}.pth"
        checkpoint_path = os.path.join(args.models_dir, checkpoint_name)
        
        if not os.path.exists(checkpoint_path):
            print(f"⚠️  Checkpoint not found: {checkpoint_path}")
            print(f"   Skipping test fold {test_split}")
            continue
        
        # Load model
        model = load_model(checkpoint_path, device)
        
        # Create test dataset
        test_indices = split_to_indices[test_split]
        test_ds = EvolutionSegmentationDataset(base_dataset, test_indices)
        test_loader = DataLoader(
            test_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0
        )
        
        fold_name = f"test_fold_{test_split}"
        
        # Run inference
        metrics = run_inference_single_fold(
            model=model,
            test_loader=test_loader,
            device=device,
            output_dir=args.output_dir,
            fold_name=fold_name,
            save_nifti=args.save_nifti,
            save_visualizations=args.save_visualizations,
            num_vis_samples=args.num_vis_samples,
        )
        
        # Store results
        result_row = {
            'test_fold': test_split,
            'val_fold': val_split,
            'num_samples': metrics['num_samples'],
            'dice_mean': metrics['dice_mean'],
            'dice_std': metrics['dice_std'],
            'dice_median': metrics['dice_median'],
            'iou_mean': metrics['iou_mean'],
            'iou_std': metrics['iou_std'],
            'iou_median': metrics['iou_median'],
        }
        all_results.append(result_row)
        
        print(f"\n📊 Results for test fold {test_split}:")
        print(f"   Samples: {metrics['num_samples']}")
        print(f"   Dice: {metrics['dice_mean']:.4f} ± {metrics['dice_std']:.4f} (median: {metrics['dice_median']:.4f})")
        print(f"   IoU:  {metrics['iou_mean']:.4f} ± {metrics['iou_std']:.4f} (median: {metrics['iou_median']:.4f})")
    
    # Save summary
    if all_results:
        results_df = pd.DataFrame(all_results)
        summary_path = os.path.join(args.output_dir, "inference_summary.csv")
        results_df.to_csv(summary_path, index=False)
        print(f"\n✅ Inference complete! Summary saved to: {summary_path}")
        
        # Print overall statistics
        print("\n" + "="*60)
        print("📈 OVERALL STATISTICS (across all folds)")
        print("="*60)
        print(f"Average Dice: {results_df['dice_mean'].mean():.4f} ± {results_df['dice_mean'].std():.4f}")
        print(f"Average IoU:  {results_df['iou_mean'].mean():.4f} ± {results_df['iou_mean'].std():.4f}")
        print(f"Total samples: {results_df['num_samples'].sum()}")
        print("="*60)
    else:
        print("\n⚠️  No results generated. Check that model checkpoints exist.")


if __name__ == "__main__":
    main()
