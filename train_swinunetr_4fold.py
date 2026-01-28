import argparse
import torch

from utils import train_swinunetr_4fold_from_csv
from main import CONFIG as MAIN_CONFIG


def main():
    ap = argparse.ArgumentParser(description="Train fold-specific SwinUNETR segmentors from 4fold_split.csv.")
    default_root = MAIN_CONFIG.get("ROOT_DIR", None)
    default_split = MAIN_CONFIG.get("FOLD_CSV", "4fold_split.csv")
    default_scan = MAIN_CONFIG.get("SCAN_NAME_STAGE2", "Scan3Wave4")

    ap.add_argument("--root_dir", default=default_root, help="Dataset root containing Scan*_FLAIR_brain and Scan*_WMH folders.")
    ap.add_argument("--split_csv", default=default_split, help="CSV with columns patient_ID and split_id/fold.")
    ap.add_argument("--scan_name", default=default_scan, help="Which timepoint to train segmentor on.")
    ap.add_argument("--models_dir", default="swinunetr_models", help="Output directory for checkpoints + summary CSV.")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_slices", type=int, default=48)
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

    if args.root_dir is None:
        raise ValueError("ROOT_DIR not set. Please provide --root_dir or set ROOT_DIR in main.py CONFIG.")

    train_swinunetr_4fold_from_csv(
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


if __name__ == "__main__":
    main()
