import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset
from flair_to_flair import Experiment1
from utils import FLAIREvolutionDataset, SwinUNetSegmentation

def run_rotated_inference():
    """
    Rotated ensemble evaluation using stratified 4-fold splits.
    Each fold's model evaluates on a designated pseudo-holdout split.
    """
    CONFIG = {
        "ROOT_DIR": "/disk/febrian/Edinburgh_Data/LBC1936",
        "FOLD_CSV": "4fold_split.csv",  # ✅ Match with main.py
        "TEST_CSV": "test_set_patients.csv",  # ✅ Added for consistency
        "CV_FOLDS": [1, 2, 3, 4],  # ✅ Match with main.py
        "DEVICE": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        "MAX_SLICES": 48,
        "BATCH_SIZE": 4,  # ✅ Match with main.py
        "K_FOLDS": 4  # ✅ Added for experiment initialization
    }

    # This mapping identifies the Model Fold and the Split it must evaluate
    # Key: Fold Index -> Value: Designated pseudo-holdout split
    rotation_map = {
        1: 4,  # Model 1 (Val 3) -> Evaluate Split 4
        2: 1,  # Model 2 (Val 4) -> Evaluate Split 1
        3: 2,  # Model 3 (Val 1) -> Evaluate Split 2
        4: 3   # Model 4 (Val 2) -> Evaluate Split 3
    }

    # ✅ Folder mapping to match actual Experiment1 output structure
    experiment_name = "flair_to_flair_baseline"  # Change this to match your experiment
    base_dir = f"exp_1_{experiment_name}"
    models_subdir = os.path.join(base_dir, "models")

    # ✅ Load the universal Stage 2 segmentor (NOT fold-specific)
    # This is a downstream task model that works on any predicted FLAIR
    segmentor_path = os.path.join(models_subdir, "wmh_segmentation_swin_unet.pth")
    
    print(f"\n🔍 Loading Stage 2 Segmentor from: {segmentor_path}")
    if not os.path.exists(segmentor_path):
        raise FileNotFoundError(
            f"❌ Segmentor model not found at {segmentor_path}\n"
            f"   Run Stage 2 segmentation training first using main.py"
        )
    
    segmentor_model = SwinUNetSegmentation(
        in_channels=1, 
        out_channels=1, 
        img_size=256, 
        feature_size=48
    ).to(CONFIG["DEVICE"])
    segmentor_model.load_state_dict(torch.load(segmentor_path, map_location=CONFIG["DEVICE"]))
    segmentor_model.eval()
    print("✅ Segmentor loaded successfully")

    split_df = pd.read_csv(CONFIG["FOLD_CSV"])    

    # Initialize Experiment to access metric reporting and 3D saving logic
    exp = Experiment1(
        experiment_number=1, 
        experiment_config={
            "name": "rotated_inference", 
            "use_wmh": True, 
            "description": "FLAIR → FLAIR rotated evaluation"
        }, 
        config=CONFIG
    )
    exp.test_patient_ids = [str(pid).zfill(4) for pid in split_df['patient_ID'].tolist()]

    all_fold_results = []

    for fold_idx, target_split in rotation_map.items():
        # ✅ Use consistent model path structure
        model_filename = f"model_fold_{fold_idx}.pth"
        model_path = os.path.join(models_subdir, model_filename)

        if not os.path.exists(model_path):
            print(f"⚠️ Warning: Checkpoint not found at {model_path}")
            continue

        print(f"\n🚀 Evaluating Fold {fold_idx} Model on Split {target_split}")
        print(f"📍 Loading weights from: {model_path}")

        # 1. Filter Patients for this specific target split
        target_pids = split_df[split_df['split_id'] == target_split]['patient_ID'].tolist()
        target_pids = [str(pid).zfill(4) for pid in target_pids]

        # 2. Setup Dataset for these specific patients (Hold-out for this model)
        dataset = FLAIREvolutionDataset(
            root_dir=CONFIG["ROOT_DIR"], 
            max_slices_per_patient=CONFIG["MAX_SLICES"],
            use_wmh=True,
            training_pairs=[("Scan1Wave2", "Scan1Wave2", 0.0)] # source identity
        )
        
        indices = [i for i, item in enumerate(dataset.index_map) 
                   if item['patient_id'] in set(target_pids)]
        
        loader = DataLoader(Subset(dataset, indices), batch_size=CONFIG["BATCH_SIZE"], shuffle=False)

        # 3. Setup Target/Ground Truth datasets for comparison
        target_datasets = {}
        for scan in ["Scan2Wave3", "Scan3Wave4", "Scan4Wave5"]:
            target_datasets[scan] = FLAIREvolutionDataset(
                root_dir=CONFIG["ROOT_DIR"],
                max_slices_per_patient=CONFIG["MAX_SLICES"],
                use_wmh=True,
                training_pairs=[(scan, scan, 0.0)],
                require_wmh_presence=False
            )

        # 4. Run evaluation with the universal segmentor
        fold_result, patient_predictions = exp._evaluate_model_with_segmentation(
            model_path=model_path,
            source_loader=loader,
            gt_loaders={},  # Not used in current implementation
            target_datasets=target_datasets,
            segmentor_model=segmentor_model  # ✅ Universal segmentor (same for all folds)
        )
        
        # 5. Inject fold index for the report later
        fold_result['fold_idx'] = fold_idx
        all_fold_results.append(fold_result)

        # 3. Generate Top-10 Visualization for THIS specific fold
        print(f"🔍 Generating Top-10 WMH visualization for Fold {fold_idx}...")
        
        tasks = {
            "Interpolation_t2": {"scan_pair": "Scan2Wave3", "time": 1.0},
            "Training_t3": {"scan_pair": "Scan3Wave4", "time": 2.0},
            "Extrapolation_t4": {"scan_pair": "Scan4Wave5", "time": 3.0},
        }

        # ✅ FIXED: Method signature - removed target_datasets parameter
        exp._visualize_top_wmh_across_test_set(
            patient_predictions, 
            tasks,  # tasks parameter contains scan_pair info
            top_n=10
        )

    # 4. Final Report: Aggregates metrics across the 4 folds
    print("\n" + "="*60)
    print("📊 AGGREGATED TEST RESULTS")
    exp._report_test_results(all_fold_results)
    
    # 5. Final Volume Analysis (Reads the saved NIfTI files)
    print("\n" + "="*60)
    print("📊 WMH VOLUME PROGRESSION ANALYSIS")
    exp._analyze_wmh_volumes_experiment3()

if __name__ == "__main__":
    run_rotated_inference()