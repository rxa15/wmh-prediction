import argparse
import os
import torch

from utils import train_swinunetr_4fold_from_csv, plot_segmentation_history

try:
    from main import CONFIG as _MAIN_CONFIG
except Exception:
    _MAIN_CONFIG = {}

DEFAULT_ROOT_DIR = _MAIN_CONFIG.get("ROOT_DIR")
DEFAULT_SPLIT_CSV = _MAIN_CONFIG.get("FOLD_CSV", "4fold_split.csv")
DEFAULT_SCAN_NAME = _MAIN_CONFIG.get("SCAN_NAME_STAGE2", "Scan1Wave2")
DEFAULT_MODELS_DIR = "swinunetr_models"
DEFAULT_EPOCHS = _MAIN_CONFIG.get("NUM_EPOCHS", 10)
DEFAULT_BATCH_SIZE = _MAIN_CONFIG.get("BATCH_SIZE", 4)
DEFAULT_LR = _MAIN_CONFIG.get("LEARNING_RATE", 1e-4)
DEFAULT_MAX_SLICES = _MAIN_CONFIG.get("MAX_SLICES", 48)


def main():
    ap = argparse.ArgumentParser(description="Train fold-specific SwinUNETR segmentors from 4fold_split.csv.")
    ap.add_argument(
        "--root_dir",
        required=DEFAULT_ROOT_DIR is None,
        default=DEFAULT_ROOT_DIR,
        help="Dataset root containing Scan*_FLAIR_brain and Scan*_WMH folders.",
    )
    ap.add_argument(
        "--split_csv",
        default=DEFAULT_SPLIT_CSV,
        help="CSV with columns patient_ID and split_id/fold.",
    )
    ap.add_argument("--scan_name", default=DEFAULT_SCAN_NAME, help="Which timepoint to train segmentor on.")
    ap.add_argument(
        "--models_dir",
        default=DEFAULT_MODELS_DIR,
        help="Output directory for checkpoints + summary CSV.",
    )
    ap.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    ap.add_argument("--batch_size", type=int, default=DEFAULT_BATCH_SIZE)
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--max_slices", type=int, default=DEFAULT_MAX_SLICES)
    ap.add_argument(
        "--val_offset",
        type=int,
        default=1,
        help="Validation split = test split rotated by this offset (default 1 => next split).",
    )
    ap.add_argument(
        "--require_wmh_presence",
        action="store_true",
        default=True,
        help="Keep only slices with WMH>0 (default: True).",
    )
    ap.add_argument(
        "--allow_empty_wmh",
        action="store_true",
        default=False,
        help="If set, includes slices even when WMH==0 (overrides --require_wmh_presence).",
    )
    ap.add_argument("--gpu", type=int, default=None, help="CUDA device index (default: auto).")
    args = ap.parse_args()

    if torch.cuda.is_available():
        if args.gpu is None:
            device = torch.device("cuda")
        else:
            device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")

    require_wmh_presence = args.require_wmh_presence and (not args.allow_empty_wmh)

    # Train models and get results with training histories
    results_df, training_histories = train_swinunetr_4fold_from_csv(
        root_dir=args.root_dir,
        split_csv_path=args.split_csv,
        scan_name=args.scan_name,
        max_slices_per_patient=args.max_slices,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        models_dir=args.models_dir,
        val_offset=args.val_offset,
        require_wmh_presence=require_wmh_presence,
    )
    
    # Generate training history plots for each fold
    print("\n" + "="*60)
    print("📊 Generating training history visualizations...")
    print("="*60)
    
    plots_dir = os.path.join(args.models_dir, "training_plots")
    for fold_idx, history in training_histories.items():
        plot_segmentation_history(history, fold_idx, save_dir=plots_dir)
    
    print(f"\n✅ Training complete! Check {args.models_dir} for:")
    print(f"   - Model checkpoints: wmh_swinunetr_test*_val*.pth")
    print(f"   - Summary CSV: swinunetr_4fold_summary.csv")
    print(f"   - Training plots: {plots_dir}/")
    print(f"\n📈 Summary of results:")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
