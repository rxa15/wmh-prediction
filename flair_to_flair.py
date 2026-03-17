# -*- coding: utf-8 -*-
"""
Experiment BL: FLAIR → FLAIR with downstream segmentation
"""

from flair_to_flair_base import BaseFlairToFlairExperiment
from utils import (
    LinearWarmupCosineAnnealingLR,
    train_epoch,
    val_epoch,
    plot_fold_history,
    plot_volume_progression,
    segment_3d_volume,
    calculate_volume_ml,
    get_ground_truth_wmh_volume,
    FLAIREvolutionDataset,
    SwinUNetSegmentation,
    load_folds_from_csv,
    BinaryDice,
)

import os
import torch
import torch.nn as nn
import torchmetrics
from torch.utils.data import DataLoader, Subset
from torch_ema import ExponentialMovingAverage
torch.cuda.empty_cache()
import pandas as pd
import numpy as np
from tqdm import tqdm
from ImageFlowNet.src.nn.imageflownet_ode import ImageFlowNetODE


class ExperimentBL(BaseFlairToFlairExperiment):
    """
    Experiment BL: FLAIR → FLAIR prediction with downstream WMH segmentation.
    
    Methodological Design:
    - Uses only L1 loss for FLAIR prediction
    - WMH segmentation is performed as a downstream evaluation task
    - use_wmh=True: WMH masks used ONLY for filtering slices with lesions
    - Model inputs/outputs: FLAIR-only (C=1) - no WMH channel in training
    - Best model selection: SSIM (structure-preserving, WMH-relevant fidelity)
    - Two-phase training: warmup (reconstruction) → full ODE training
    - Training metrics: evaluated on full training set (not sampled)
    """
    
    require_wmh_presence = False
    use_wmh_for_stage1_dataset = False
    stage1_max_slices_per_patient = 12
    stage1_dataset_kwargs = {
        "slice_selection_mode": "random_valid",
        "valid_slice_mode": "wmh",
        "random_seed": 42,
    }
    run_title = "FLAIR → FLAIR (L1 only)"
    
    # def _diagnose_wmh(self, dataset):
    #     print("\n================ WMH DIAGNOSTIC ================")

    #     num_total = len(dataset)
    #     num_with_wmh = 0
    #     patients_with_wmh = set()

    #     for i in range(num_total):
    #         item = dataset[i]
    #         target = item["target"]

    #         # If dataset loads only FLAIR (1 channel), skip
    #         if target.shape[0] == 1:
    #             continue

    #         wmh_mask = target[1]  # Channel 1 = WMH
    #         if wmh_mask.sum() > 0:
    #             num_with_wmh += 1
    #             patients_with_wmh.add(item["patient_id"])

    #     print(f"Total slices              : {num_total}")
    #     print(f"Slices with WMH > 0       : {num_with_wmh}")
    #     print(f"Patients with WMH slices  : {len(patients_with_wmh)}")
    #     print(f"List of patients          : {sorted(list(patients_with_wmh))}")

    #     if num_with_wmh == 0:
    #         print("⚠️  NO WMH SLICES FOUND!")
    #         print("Possible reasons:")
    #         print(" - use_wmh=False somewhere")
    #         print(" - WMH paths missing in dataset")
    #         print(" - Slices with WMH got skipped due to slice selection (14:)")
    #         print(" - Wrong scan pairs (t1->t3 has no WMH available)")
    #         print("=================================================\n")
    #     else:
    #         print("✅ WMH found in dataset.")
    #         print("=================================================\n")
    
    def run(self):
        """Execute the full Experiment BL pipeline."""
        print("\n" + "="*60)
        print("Starting Experiment BL: FLAIR → FLAIR")
        print("="*60 + "\n")
        
        # Stage 1: Train ImageFlowNet models
        predicted_flair_dir, ground_truth_wmh_dir = self._stage1_train_imageflownet()
        
        # Stage 2: WMH Segmentation from predicted FLAIR
        if predicted_flair_dir and ground_truth_wmh_dir:
            self._stage2_wmh_segmentation(predicted_flair_dir, ground_truth_wmh_dir)
        else:
            print("[Stage 2] Skipped because Stage 1 did not complete successfully.")
    
    def _stage1_train_imageflownet(self):
        """Train ImageFlowNet models using cross-validation."""
        print("="*60)
        print("✅ STAGE 1: ImageFlowNet Training")
        print("="*60)
        
        print("Initializing dataset with custom training pairs (all temporal pairs)...")
        full_dataset = self._create_stage1_dataset()
        folds_dict = self._load_folds_dict()
        
        # Full 4-fold CV with train/val/test = 2/1/1 folds
        print(f"\n📈 Starting {self.config['K_FOLDS']}-Fold Cross-Validation Training (2 train / 1 val / 1 test)...")

        cv_folds = list(self.config["CV_FOLDS"])
        val_offset = int(self.config.get("VAL_OFFSET", 1))

        # Train one model per test fold.
        for test_fold in cv_folds:
            val_fold = cv_folds[(cv_folds.index(test_fold) + val_offset) % len(cv_folds)]
            train_folds = [f for f in cv_folds if f not in (test_fold, val_fold)]
            self._train_fold(test_fold, val_fold, train_folds, full_dataset, folds_dict)

        # Final Evaluation (per-fold test splits)
        return self._evaluate_on_test_set(full_dataset, folds_dict, val_offset=val_offset)
    
    def _train_fold(self, test_fold_idx, val_fold_idx, train_fold_idxs, full_dataset, folds_dict):
        """Train a model on a single fold (2 train / 1 val / 1 test)."""
        print(f"\n{'='*50}")
        print(f"K-Fold Run: Test Fold {test_fold_idx} | Val Fold {val_fold_idx} | Train Folds {train_fold_idxs}")
        print(f"{'='*50}\n")

        raw_val_pids = folds_dict[val_fold_idx]
        raw_train_pids = [pid for f_idx in train_fold_idxs for pid in folds_dict[f_idx]]

        val_pids = raw_val_pids[:self.config["MAX_PATIENTS_PER_FOLD"]]
        train_pids = raw_train_pids[: self.config["MAX_PATIENTS_PER_FOLD"] * len(train_fold_idxs)]
                
        print(f"Training patients:   {len(train_pids)}")
        print(f"Validation patients: {len(val_pids)}")
        
        # Data indices
        train_indices = [i for i, item in enumerate(full_dataset.index_map) 
                        if item['patient_id'] in set(train_pids)]
        val_indices = [i for i, item in enumerate(full_dataset.index_map) 
                      if item['patient_id'] in set(val_pids)]
        
        # DataLoaders
        train_loader = DataLoader(
            Subset(full_dataset, train_indices),
            batch_size=self.config["BATCH_SIZE"],
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            Subset(full_dataset, val_indices),
            batch_size=self.config["BATCH_SIZE"],
            shuffle=False,
            num_workers=0
        )
        
        # Initialize model
        model = ImageFlowNetODE(
            device=self.config["DEVICE"],
            in_channels=1,
            ode_location='bottleneck',
            contrastive=True
        ).to(self.config["DEVICE"])
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config["LEARNING_RATE"])
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.config["NUM_EPOCHS"] // 10,
            max_epochs=self.config["NUM_EPOCHS"]
        )
        ema = ExponentialMovingAverage(model.parameters(), decay=0.9)
        recon_loss = nn.MSELoss()
        max_time_delta = max(time_delta for _, _, time_delta in self.training_pairs)
        t_multiplier = self.config.get("ODE_MAX_T", max_time_delta) / max_time_delta
        self.config["T_MULTIPLIER"] = t_multiplier
        
        # Two-phase training: warmup with reconstruction, then full ODE training
        warmup_epochs = self.config.get("WARMUP_EPOCHS", self.config["NUM_EPOCHS"] // 10)
        
        # Best model selection: use SSIM (structure-preserving) for WMH-relevant fidelity
        best_val_ssim = 0.0
        model_save_path = self.get_model_path(test_fold_idx)
        
        history = {
            'train_recon_loss': [],
            'train_pred_loss': [],
            'train_recon_psnr': [],   
            'train_pred_psnr': [],    
            'train_delta_psnr': [],
            'train_pred_ssim': [],
            'val_recon_psnr': [],
            'val_pred_psnr': [],
            'val_delta_psnr': [],
            'val_pred_ssim': []
        }
        
        # Training loop
        for epoch in range(self.config["NUM_EPOCHS"]):
            # Phase 1 (warmup): focus on reconstruction stability
            # Phase 2: full ODE training with time-dependent prediction
            use_ode_training = epoch >= warmup_epochs
            
            avg_recon_loss, avg_pred_loss = train_epoch(
                model, train_loader, optimizer, ema, recon_loss,
                self.config["DEVICE"], epoch, use_ode_training,
                self.config["NUM_EPOCHS"], self.config["CONTRASTIVE_COEFF"],
                smoothness_coeff=self.config.get("SMOOTHNESS_COEFF", 0.0),
                latent_coeff=self.config.get("LATENT_COEFF", 0.0),
                invariance_coeff=self.config.get("INVARIANCE_COEFF", 0.0),
                no_l2=self.config.get("NO_L2", False),
                t_multiplier=t_multiplier,
                noise_max_intensity=self.config.get("NOISE_MAX_INTENSITY", 0.1),
            )
            
            # Validation metrics
            with ema.average_parameters():
                val_recon_psnr, val_pred_psnr = val_epoch(
                    model, val_loader, self.config["DEVICE"], t_multiplier=t_multiplier
                )
                # Calculate SSIM for WMH-structure evaluation
                val_pred_ssim = self._compute_ssim(model, val_loader)
            
            # Training metrics: evaluate on FULL training set (not sampled)
            # Evaluate every N epochs to save time
            train_eval_frequency = self.config.get("TRAIN_EVAL_EVERY", 5)
            if (epoch + 1) % train_eval_frequency == 0 or epoch == 0:
                with torch.no_grad(), ema.average_parameters():
                    train_eval_loader = DataLoader(
                        Subset(full_dataset, train_indices),
                        batch_size=self.config["BATCH_SIZE"],
                        shuffle=False,
                        num_workers=0
                    )
                    train_recon_psnr, train_pred_psnr = val_epoch(
                        model, train_eval_loader, self.config["DEVICE"], t_multiplier=t_multiplier
                    )
                    train_pred_ssim = self._compute_ssim(model, train_eval_loader)
            
            # Assert FLAIR-only inputs/outputs (thesis-defensible)
            if epoch == 0:  # Check once at start
                sample_batch = next(iter(train_loader))
                assert sample_batch["source"].shape[1] == 1, "This experiment must use FLAIR-only input (C=1)"
                assert sample_batch["target"].shape[1] == 1, "This experiment must use FLAIR-only target (C=1)"
                print("✅ Verified: FLAIR-only inputs (C=1) - WMH used only for slice filtering, not as model input")
            
            # Compute delta PSNR for history
            val_delta_psnr = val_pred_psnr - val_recon_psnr
            train_delta_psnr = train_pred_psnr - train_recon_psnr

            history['train_recon_loss'].append(avg_recon_loss)
            history['train_pred_loss'].append(avg_pred_loss)
            history['train_recon_psnr'].append(train_recon_psnr)  
            history['train_pred_psnr'].append(train_pred_psnr)
            history['train_delta_psnr'].append(train_delta_psnr)
            history['train_pred_ssim'].append(train_pred_ssim)
            history['val_recon_psnr'].append(val_recon_psnr)
            history['val_pred_psnr'].append(val_pred_psnr)
            history['val_delta_psnr'].append(val_delta_psnr)
            history['val_pred_ssim'].append(val_pred_ssim)
            
            phase = "Warmup" if epoch < warmup_epochs else "ODE Training"
            print(f"""Epoch {epoch+1} [{phase}]: Train Pred PSNR={train_pred_psnr:.4f}, SSIM={train_pred_ssim:.4f} | Val Pred PSNR={val_pred_psnr:.4f}, SSIM={val_pred_ssim:.4f}""")
            
            # Save best model by SSIM (structure-preserving for WMH)
            if val_pred_ssim > best_val_ssim:
                best_val_ssim = val_pred_ssim
                torch.save(model.state_dict(), model_save_path)
                print(f"✅ Val SSIM improved to {val_pred_ssim:.4f}. Model saved to {model_save_path}")
            
            scheduler.step()
        
        # Plot training history
        plot_fold_history(history, val_fold_idx, self.plots_dir)
        
        # Save training history to CSV
        history_df = pd.DataFrame({
            'epoch': range(1, len(history['train_recon_loss']) + 1),
            'train_recon_loss': history['train_recon_loss'],
            'train_pred_loss': history['train_pred_loss'],
            'train_recon_psnr': history['train_recon_psnr'],
            'train_pred_psnr': history['train_pred_psnr'],
            'train_delta_psnr': history['train_delta_psnr'],
            'train_pred_ssim': history['train_pred_ssim'],
            'val_recon_psnr': history['val_recon_psnr'],
            'val_pred_psnr': history['val_pred_psnr'],
            'val_delta_psnr': history['val_delta_psnr'],
            'val_pred_ssim': history['val_pred_ssim'],
        })
        history_csv_path = os.path.join(self.results_dir, f"training_history_fold_{test_fold_idx}.csv")
        history_df.to_csv(history_csv_path, index=False)
        print(f"📊 Training history saved to {history_csv_path}")
    
    def _evaluate_on_test_set(self, full_dataset, folds_dict, val_offset=1):
        """Evaluate ensemble predictions on each fold's held-out test patients."""
        print("\n" + "=" * 60)
        print("Ensemble evaluation on per-fold test splits")
        print("=" * 60)

        all_stage1_model_paths = [
            self.get_model_path(fold_idx)
            for fold_idx in self.config["CV_FOLDS"]
            if os.path.exists(self.get_model_path(fold_idx))
        ]
        if not all_stage1_model_paths:
            print("No trained Stage 1 models found to evaluate.")
            return None, None

        _, loaded_segmentors = self._load_fold_specific_segmentors()
        source_dataset, target_datasets = self._create_ensemble_eval_datasets()

        all_results = []
        for test_fold in self.config["CV_FOLDS"]:
            test_pids = set(folds_dict[test_fold])
            eligible_model_paths = self._get_leakage_free_ensemble_model_paths(test_fold)
            if not eligible_model_paths:
                print(f"⚠️ No leakage-free ensemble members available for test fold {test_fold}. Skipping.")
                continue

            eligible_model_names = [os.path.basename(path) for path in eligible_model_paths]
            print(
                f"Test fold {test_fold}: using {len(eligible_model_paths)} leakage-free ensemble member(s): "
                f"{eligible_model_names}"
            )

            source_test_indices = [
                i for i, item in enumerate(source_dataset.index_map)
                if item["patient_id"] in test_pids
            ]
            source_loader = DataLoader(
                Subset(source_dataset, source_test_indices),
                batch_size=self.config["BATCH_SIZE"],
                shuffle=False,
                num_workers=0,
            )

            output_prefix = f"ensemble_test_fold_{test_fold}"
            fold_result, patient_predictions = self._evaluate_ensemble_with_segmentation(
                model_paths=eligible_model_paths,
                source_loader=source_loader,
                target_datasets=target_datasets,
                segmentor_model=loaded_segmentors[test_fold],
                output_prefix=output_prefix,
            )
            fold_result["fold"] = test_fold
            all_results.append(fold_result)

        if not all_results:
            return None, None

        self._report_test_results(all_results)
        ground_truth_wmh_dir = os.path.join(self.config["ROOT_DIR"], "Scan3Wave4_WMH")
        return self.results_dir, ground_truth_wmh_dir
    
    def _create_gt_lookup(self, gt_datasets):
        """Create a lookup dictionary for efficient ground truth retrieval."""
        gt_lookup = {}
        for scan_name, dataset in gt_datasets.items():
            gt_lookup[scan_name] = {}
            for idx, item in enumerate(dataset.index_map):
                p_id = item['patient_id']
                s_idx = item['slice_idx']
                gt_lookup[scan_name][(p_id, s_idx)] = idx
        return gt_lookup

    def _load_fold_specific_segmentors(self):
        """Load pretrained fold-specific WMH segmentors."""
        swinunetr_models_dir = self.config.get("SWINUNETR_MODELS_DIR", "swinunetr_models")
        if not os.path.exists(swinunetr_models_dir):
            raise FileNotFoundError(
                f"Pretrained SwinUNETR models directory not found: {swinunetr_models_dir}"
            )

        fold_to_model = {}
        loaded_models = {}
        cv_folds = list(self.config["CV_FOLDS"])
        val_offset = int(self.config.get("VAL_OFFSET", 1))

        for test_fold in cv_folds:
            val_fold = cv_folds[(cv_folds.index(test_fold) + val_offset) % len(cv_folds)]
            model_filename = f"wmh_swinunetr_test{test_fold}_val{val_fold}.pth"
            model_path = os.path.join(swinunetr_models_dir, model_filename)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Missing pretrained segmentation model: {model_path}")

            model = SwinUNetSegmentation().to(self.config["DEVICE"])
            model.load_state_dict(torch.load(model_path, map_location=self.config["DEVICE"]))
            model.eval()
            fold_to_model[test_fold] = model_path
            loaded_models[test_fold] = model

        return fold_to_model, loaded_models

    def _create_ensemble_eval_datasets(self):
        """Create source and target datasets for fold-wise ensemble evaluation."""
        source_dataset = FLAIREvolutionDataset(
            root_dir=self.config["ROOT_DIR"],
            max_slices_per_patient=self.config["MAX_SLICES"],
            use_wmh=False,
            training_pairs=[("Scan1Wave2", "Scan1Wave2", 0.0)],
            require_wmh_presence=False,
        )

        target_datasets = {}
        for target_scan in ["Scan2Wave3", "Scan3Wave4", "Scan4Wave5"]:
            target_datasets[target_scan] = FLAIREvolutionDataset(
                root_dir=self.config["ROOT_DIR"],
                max_slices_per_patient=self.config["MAX_SLICES"],
                use_wmh=True,
                training_pairs=[(target_scan, target_scan, 0.0)],
                require_wmh_presence=False,
            )

        return source_dataset, target_datasets

    def _get_leakage_free_ensemble_model_paths(self, eval_test_fold):
        """Return Stage 1 model paths that did not train on the evaluated test fold."""
        cv_folds = list(self.config["CV_FOLDS"])
        val_offset = int(self.config.get("VAL_OFFSET", 1))
        eligible_model_paths = []

        for model_test_fold in cv_folds:
            model_path = self.get_model_path(model_test_fold)
            if not os.path.exists(model_path):
                continue

            model_val_fold = cv_folds[(cv_folds.index(model_test_fold) + val_offset) % len(cv_folds)]
            model_train_folds = [f for f in cv_folds if f not in (model_test_fold, model_val_fold)]

            if eval_test_fold in model_train_folds:
                continue

            eligible_model_paths.append(model_path)

        return eligible_model_paths

    def _evaluate_ensemble_with_segmentation(
        self,
        model_paths,
        source_loader,
        target_datasets,
        segmentor_model,
        output_prefix,
    ):
        """Evaluate an ensemble of Stage 1 models with downstream WMH segmentation."""
        models = []
        for model_path in model_paths:
            model = ImageFlowNetODE(
                device=self.config["DEVICE"],
                in_channels=1,
                ode_location='bottleneck',
                contrastive=True,
            ).to(self.config["DEVICE"])
            model.load_state_dict(torch.load(model_path, map_location=self.config["DEVICE"]))
            model.eval()
            models.append(model)

        tasks = {
            "Interpolation_t2": {"scan_pair": "Scan2Wave3", "time": 3.0},
            "Training_t3": {"scan_pair": "Scan3Wave4", "time": 6.0},
            "Extrapolation_t4": {"scan_pair": "Scan4Wave5", "time": 9.0},
        }
        metrics = {
            name: {
                "psnr_flair": torchmetrics.PeakSignalNoiseRatio(data_range=1.0).to(self.config["DEVICE"]),
                "ssim_flair": torchmetrics.StructuralSimilarityIndexMeasure(data_range=1.0).to(self.config["DEVICE"]),
                "dice_wmh": BinaryDice(threshold=0.5).to(self.config["DEVICE"]),
            }
            for name in tasks
        }
        gt_lookup = self._create_gt_lookup(target_datasets)
        patient_predictions = {task_name: {} for task_name in tasks}
        t_multiplier = self.config.get("T_MULTIPLIER", 1.0)

        with torch.no_grad():
            for source_batch in tqdm(source_loader, desc=f"Evaluating {output_prefix}"):
                source_flair = source_batch["source"].to(self.config["DEVICE"])
                patient_ids, slice_indices = source_batch["patient_id"], source_batch["slice_idx"]

                for task_name, task_info in tasks.items():
                    t = torch.tensor([task_info["time"] * t_multiplier], device=self.config["DEVICE"])
                    ensemble_pred_flair = torch.stack(
                        [model(source_flair, t=t) for model in models],
                        dim=0,
                    ).mean(dim=0)

                    pred_flair_normalized = torch.zeros_like(ensemble_pred_flair)
                    for sample_idx in range(ensemble_pred_flair.shape[0]):
                        sample = ensemble_pred_flair[sample_idx]
                        min_val = sample.min()
                        max_val = sample.max()
                        if max_val > min_val:
                            pred_flair_normalized[sample_idx] = (sample - min_val) / (max_val - min_val)
                        else:
                            pred_flair_normalized[sample_idx] = sample
                    pred_flair_normalized = torch.clamp(pred_flair_normalized, 0.0, 1.0)
                    pred_wmh_prob = torch.sigmoid(segmentor_model(pred_flair_normalized))

                    gt_pair_name = task_info["scan_pair"]
                    target_flair_valid = []
                    target_wmh_valid = []
                    pred_flair_valid = []
                    pred_wmh_prob_valid = []

                    for j in range(len(patient_ids)):
                        p_id = patient_ids[j]
                        p_id_str = str(p_id.item()).zfill(4) if isinstance(p_id, torch.Tensor) else str(p_id).zfill(4)
                        s_idx = slice_indices[j].item()
                        gt_idx = gt_lookup[gt_pair_name].get((p_id_str, s_idx))
                        if gt_idx is None:
                            continue

                        gt_sample = target_datasets[gt_pair_name][gt_idx]
                        tgt = gt_sample["target"].to(self.config["DEVICE"])
                        target_flair_valid.append(tgt[0:1, :, :])
                        if tgt.shape[0] > 1:
                            target_wmh_valid.append(tgt[1:2, :, :].int())
                        else:
                            target_wmh_valid.append(torch.zeros_like(tgt[0:1, :, :]).int())

                        pred_flair_valid.append(ensemble_pred_flair[j, 0:1, :, :])
                        pred_wmh_prob_valid.append(pred_wmh_prob[j, 0:1, :, :])

                        if p_id_str not in patient_predictions[task_name]:
                            patient_predictions[task_name][p_id_str] = {}
                        patient_predictions[task_name][p_id_str][s_idx] = {
                            "flair": ensemble_pred_flair[j, 0].cpu().numpy(),
                            "wmh": pred_wmh_prob[j, 0].cpu().numpy(),
                        }

                    if not target_flair_valid:
                        continue

                    target_flair_batch = torch.stack(target_flair_valid)
                    pred_flair_batch = torch.stack(pred_flair_valid)
                    target_wmh_batch = torch.stack(target_wmh_valid)
                    pred_wmh_prob_batch = torch.stack(pred_wmh_prob_valid)

                    metrics[task_name]["psnr_flair"].update(pred_flair_batch, target_flair_batch)
                    metrics[task_name]["ssim_flair"].update(pred_flair_batch, target_flair_batch)
                    metrics[task_name]["dice_wmh"].update(pred_wmh_prob_batch, target_wmh_batch)

        final_results = {"ensemble_name": output_prefix}
        for task_name in tasks:
            final_results[task_name] = {
                "PSNR": metrics[task_name]["psnr_flair"].compute().item(),
                "SSIM": metrics[task_name]["ssim_flair"].compute().item(),
                "Dice": metrics[task_name]["dice_wmh"].compute().item(),
            }

        self._save_3d_predictions_with_prefix(patient_predictions, tasks, output_prefix)
        return final_results, patient_predictions
    
    def _save_3d_predictions_with_prefix(self, patient_predictions, tasks, output_prefix):
        """Save 3D FLAIR and WMH predictions using a caller-provided prefix."""
        import nibabel as nib

        original_scans_dir = os.path.join(self.config["ROOT_DIR"], "Scan1Wave2_FLAIR_brain")

        for task_name, task_info in tasks.items():
            predictions_by_patient = patient_predictions[task_name]
            gt_pair_name = task_info["scan_pair"]

            flair_save_dir = os.path.join(self.results_dir, f"{output_prefix}_Pred_{gt_pair_name}_FLAIR_3D")
            wmh_save_dir = os.path.join(self.results_dir, f"{output_prefix}_Pred_{gt_pair_name}_WMH_3D")
            os.makedirs(flair_save_dir, exist_ok=True)
            os.makedirs(wmh_save_dir, exist_ok=True)

            for patient_id, slices in predictions_by_patient.items():
                if not slices:
                    continue

                max_slice_idx = max(slices.keys())
                h, w = next(iter(slices.values()))["flair"].shape
                flair_volume = np.zeros((h, w, max_slice_idx + 1), dtype=np.float32)
                wmh_volume = np.zeros((h, w, max_slice_idx + 1), dtype=np.float32)

                for slice_idx, pred_data in slices.items():
                    flair_volume[:, :, slice_idx] = pred_data["flair"]
                    wmh_volume[:, :, slice_idx] = pred_data["wmh"]

                affine = np.eye(4)
                try:
                    full_prefix = f"LBC36{patient_id.zfill(4)}"
                    original_file = next(f for f in os.listdir(original_scans_dir) if f.startswith(full_prefix))
                    affine = nib.load(os.path.join(original_scans_dir, original_file)).affine
                except Exception:
                    pass

                nib.save(
                    nib.Nifti1Image(flair_volume, affine),
                    os.path.join(flair_save_dir, f"{patient_id}_predicted_flair_3D.nii.gz"),
                )
                nib.save(
                    nib.Nifti1Image(wmh_volume, affine),
                    os.path.join(wmh_save_dir, f"{patient_id}_predicted_wmh_3D.nii.gz"),
                )

    def _report_test_results(self, all_results):
        """Aggregate and report test set results."""
        print("\n" + "="*60)
        print("📊 Final Test Set Results (Mean ± Std Dev)")
        print("="*60)
        
        tasks = ["Interpolation_t2", "Training_t3", "Extrapolation_t4"]
        task_labels = {
            "Interpolation_t2": "Interpolation (t1→t2, Δt=1.0)",
            "Training_t3": "Training (t1→t3, Δt=2.0)",
            "Extrapolation_t4": "Extrapolation (t1→t4, Δt=3.0)"
        }
        
        for task in tasks:
            psnrs = [r[task]['PSNR'] for r in all_results]
            ssims = [r[task]['SSIM'] for r in all_results]
            dices = [r[task]['Dice'] for r in all_results]
            
            print(f"\n{task_labels[task]}:")
            print(f"  FLAIR PSNR: {np.nanmean(psnrs):.4f} ± {np.nanstd(psnrs):.4f} dB")
            print(f"  FLAIR SSIM: {np.nanmean(ssims):.4f} ± {np.nanstd(ssims):.4f}")
            print(f"  WMH Dice:   {np.nanmean(dices):.4f} ± {np.nanstd(dices):.4f}")
        
        # Save to CSV
        test_results_data = []
        for i, result in enumerate(all_results):
            fold_idx = self.config["CV_FOLDS"][i]
            for task in tasks:
                test_results_data.append({
                    'fold': fold_idx,
                    'task': task,
                    'psnr_flair': result[task]['PSNR'],
                    'ssim_flair': result[task]['SSIM'],
                    'dice_wmh': result[task]['Dice']
                })
        
        test_results_df = pd.DataFrame(test_results_data)
        
        # Add summary statistics
        summary_rows = []
        for task in tasks:
            task_data = test_results_df[test_results_df['task'] == task]
            summary_rows.append({
                'fold': 'mean',
                'task': task,
                'psnr_flair': task_data['psnr_flair'].mean(),
                'ssim_flair': task_data['ssim_flair'].mean(),
                'dice_wmh': task_data['dice_wmh'].mean()
            })
            summary_rows.append({
                'fold': 'std',
                'task': task,
                'psnr_flair': task_data['psnr_flair'].std(),
                'ssim_flair': task_data['ssim_flair'].std(),
                'dice_wmh': task_data['dice_wmh'].std()
            })
        
        test_results_df = pd.concat([test_results_df, pd.DataFrame(summary_rows)], ignore_index=True)
        
        csv_path = os.path.join(self.results_dir, "test_set_evaluation_results.csv")
        test_results_df.to_csv(csv_path, index=False)
        print(f"\n📊 Test results saved to {csv_path}")
        print("="*60)
    
    def _analyze_wmh_volumes_with_pretrained_models(self, predicted_flair_base_dir, gt_wmh_dirs, 
                                                    time_points, fold_to_model, folds_dict):
        """
        Analyze WMH volume progression using fold-specific pretrained models.
        
        Args:
            predicted_flair_base_dir: Base directory containing fold-specific prediction folders
            gt_wmh_dirs: Dictionary mapping time points to ground truth WMH directories
            time_points: List of time point names (e.g., ['Scan1Wave2', 'Scan2Wave3', ...])
            fold_to_model: Dictionary mapping fold numbers to pretrained model paths
            folds_dict: Dictionary mapping fold numbers to patient IDs
            
        Returns:
            volume_results: Dictionary with patient-level volume progression data
        """
        import nibabel as nib
        from utils import calculate_volume_ml, get_ground_truth_wmh_volume, segment_3d_volume
        
        print("\n📊 Analyzing WMH volumes with fold-specific pretrained models...")
        
        volume_results = {}
        device = self.config["DEVICE"]
        
        # Build a reverse map: patient_id -> (fold, model_path)
        patient_to_model = {}
        for fold_num, patient_ids in folds_dict.items():
            if fold_num in fold_to_model:
                model_path = fold_to_model[fold_num]
                for patient_id in patient_ids:
                    patient_to_model[patient_id] = (fold_num, model_path)
        
        all_patients = set(patient_to_model.keys())
        print(f"Found {len(all_patients)} patients with fold assignments")
        
        # Load models only once per fold
        loaded_models = {}
        for fold_num, model_path in fold_to_model.items():
            print(f"Loading pretrained model for fold {fold_num}: {os.path.basename(model_path)}")
            model = SwinUNetSegmentation().to(device)
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            loaded_models[fold_num] = model
        
        # Process each patient
        for patient_id in tqdm(all_patients, desc="Processing patients"):
            # Determine which model to use for this patient
            if patient_id not in patient_to_model:
                print(f"⚠️ Patient {patient_id} not found in any fold, skipping...")
                continue
            
            fold_num, model_path = patient_to_model[patient_id]
            seg_model = loaded_models[fold_num]
            
            patient_volumes = {'predicted': [], 'ground_truth': [], 'time_points': []}
            
            for time_point in time_points:
                pred_dir = os.path.join(
                    predicted_flair_base_dir,
                    f"ensemble_test_fold_{fold_num}_Pred_{time_point}_FLAIR_3D",
                )
                if not os.path.exists(pred_dir):
                    continue

                pred_file = f"{patient_id}_predicted_flair_3D.nii.gz"
                pred_flair_path = os.path.join(pred_dir, pred_file)
                
                if not os.path.exists(pred_flair_path):
                    continue
                
                try:
                    # Load predicted FLAIR
                    pred_flair_nii = nib.load(pred_flair_path)
                    pred_flair_volume = pred_flair_nii.get_fdata(dtype=np.float32)
                    pred_affine = pred_flair_nii.affine
                    
                    # Segment using fold-specific model
                    pred_wmh_volume = segment_3d_volume(seg_model, pred_flair_volume, device)
                    pred_wmh_ml = calculate_volume_ml(pred_wmh_volume, affine=pred_affine)
                    
                    # Load ground truth
                    gt_wmh_volume, gt_affine = get_ground_truth_wmh_volume(gt_wmh_dirs[time_point], patient_id)
                    gt_wmh_ml = calculate_volume_ml(gt_wmh_volume, affine=gt_affine) if gt_wmh_volume is not None else 0
                    
                    patient_volumes['predicted'].append(pred_wmh_ml)
                    patient_volumes['ground_truth'].append(gt_wmh_ml)
                    patient_volumes['time_points'].append(time_point)
                    
                except Exception as e:
                    print(f"⚠️ Error processing patient {patient_id} at {time_point}: {e}")
                    continue
            
            if patient_volumes['predicted']:
                volume_results[patient_id] = patient_volumes
        
        print(f"✅ Successfully analyzed {len(volume_results)} patients")
        return volume_results
    
    def _stage2_wmh_segmentation(self, pred_flair_dir, wmh_gt_dir):
        """Run Stage 2: WMH Segmentation using pretrained fold-specific models."""
        print("\n" + "="*60)
        print("Starting Stage 2 (WMH Segmentation with Pretrained Models)")
        print("="*60)
        
        if not os.path.exists(wmh_gt_dir):
            print(f"[Stage 2] Directory not found: {wmh_gt_dir}")
            return
        
        swinunetr_models_dir = self.config.get("SWINUNETR_MODELS_DIR", "swinunetr_models")
        print(f"✅ Using pretrained SwinUNETR models from: {swinunetr_models_dir}")
        print("Note: Skipping retraining as pretrained models are available.\n")
        
        # Load fold assignments to match predictions with correct models
        fold_csv = self.config["FOLD_CSV"]
        folds_dict = load_folds_from_csv(fold_csv)
        val_offset = int(self.config.get("VAL_OFFSET", 1))
        cv_folds = list(self.config["CV_FOLDS"])
        
        fold_to_model, _ = self._load_fold_specific_segmentors()
        
        print(f"\n📊 Total pretrained models available: {len(fold_to_model)}")
        
        # Note: The actual segmentation will be performed in analyze_wmh_volume_progression
        # We just need to ensure the models are available
        
        # Volume progression analysis
        print("\n" + "="*60)
        print("Performing WMH Volume Progression Analysis")
        print("="*60 + "\n")
        
        time_points = ['Scan1Wave2', 'Scan2Wave3', 'Scan3Wave4', 'Scan4Wave5']
        gt_wmh_dirs = {
            tp: os.path.join(self.config["ROOT_DIR"], f"{tp}_WMH")
            for tp in time_points
        }
        
        # Check for missing directories
        missing = [tp for tp, d in gt_wmh_dirs.items() if not os.path.exists(d)]
        if missing:
            print(f"⚠️ Missing directories for: {missing}")
            return
        
        # Analyze volumes with fold-specific models
        # predicted_flair_base_dir should contain subfolders like "model_fold_X_Pred_ScanYWaveZ_3D"
        # We need to use the correct pretrained model for each fold's predictions
        volume_results = self._analyze_wmh_volumes_with_pretrained_models(
            self.results_dir,  # Base dir containing prediction folders
            gt_wmh_dirs,
            time_points,
            fold_to_model,  # Map of fold -> pretrained model path
            folds_dict  # Patient assignments to folds
        )
        
        if volume_results:
            plot_volume_progression(volume_results, self.get_plots_path("volume_progression.png"))
            
            # Save results to CSV
            df_results = []
            for patient_id, volumes in volume_results.items():
                for i, time_point in enumerate(volumes['time_points']):
                    df_results.append({
                        'patient_id': patient_id,
                        'time_point': time_point,
                        'predicted_wmh_ml': volumes['predicted'][i],
                        'ground_truth_wmh_ml': volumes['ground_truth'][i],
                        'volume_error_ml': volumes['predicted'][i] - volumes['ground_truth'][i]
                    })
            
            df = pd.DataFrame(df_results)
            csv_path = self.get_results_path(f"wmh_volume_progression_{self.name}.csv")
            df.to_csv(csv_path, index=False)
            print(f"✅ Volume progression results saved to {csv_path}")
            
            # Print summary
            errors = [row['volume_error_ml'] for row in df_results]
            if errors:
                print(f"\n📊 Volume Analysis Summary:")
                print(f"   Mean Error: {np.mean(errors):.2f} +/- {np.std(errors):.2f} ml")
                print(f"   Min Error: {np.min(errors):.2f} ml")
                print(f"   Max Error: {np.max(errors):.2f} ml")


# ============================================================
# === STANDALONE EXECUTION ===
# ============================================================

if __name__ == "__main__":
    """
    Run this experiment directly without going through main.py
    Usage: python flair_to_flair.py
    """
    print("\n" + "="*70)
    print("🧪 Running Experiment BL: FLAIR → FLAIR (Standalone Mode)")
    print("="*70 + "\n")
    
    # Import config from main.py (reuse the same config)
    from main import CONFIG as MAIN_CONFIG
    CONFIG = MAIN_CONFIG
    
    print(f"Using device: {CONFIG['DEVICE']}")
    
    # Experiment configuration
    experiment_config = {
        "name": "flair_to_flair_bl",
        "description": "FLAIR -> FLAIR baseline (two-stage: prediction then segmentation)",
        "use_wmh": True,
        "class": ExperimentBL
    }
    
    # Run experiment
    experiment = ExperimentBL(
        experiment_number=1,
        experiment_config=experiment_config,
        config=CONFIG
    )
    experiment.run()
    
    print("\n" + "="*70)
    print("✅ EXPERIMENT COMPLETE")
    print("="*70 + "\n")
