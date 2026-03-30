# -*- coding: utf-8 -*-

import os

from base import BaseExperiment
from utils import (
    ALL_FLAIR_TEMPORAL_PAIRS,
    FLAIREvolutionDataset,
    compute_prediction_ssim,
    load_folds_from_csv,
    load_patient_ids_from_csv,
)


class BaseFlairToFlairExperiment(BaseExperiment):
    use_held_out_test_csv = False
    require_wmh_presence = False
    use_wmh_for_stage1_dataset = None
    training_pairs = ALL_FLAIR_TEMPORAL_PAIRS
    stage1_max_slices_per_patient = None
    stage1_dataset_kwargs = {}
    run_title = "FLAIR → FLAIR"
    stage1_title = "ImageFlowNet Training"

    def __init__(self, experiment_number, experiment_config, config):
        super().__init__(experiment_number, experiment_config)
        self.config = config

        if self.use_held_out_test_csv:
            test_csv = self.config["TEST_CSV"]
            if test_csv is None or not os.path.exists(test_csv):
                raise FileNotFoundError(f"Test CSV not found at {test_csv}")
            self.test_patient_ids = self._load_patient_ids(test_csv, column="patient_ID")
            print(f"Loaded {len(self.test_patient_ids)} explicit test patients from {test_csv}")
        else:
            self.test_patient_ids = None

    def _load_patient_ids(self, csv_path, column="patient_ID"):
        return load_patient_ids_from_csv(csv_path, column=column, normalize=not self.use_held_out_test_csv)

    def _compute_ssim(self, model, loader):
        return compute_prediction_ssim(
            model,
            loader,
            self.config["DEVICE"],
            t_multiplier=self.config.get("T_MULTIPLIER", 1.0),
        )

    def _get_stage1_time_deltas(self, dataset=None):
        """Return the time deltas used by Stage 1 training."""
        if self.training_pairs:
            return [float(time_delta) for _, _, time_delta in self.training_pairs]

        if dataset is not None:
            deltas = {
                float(item["time_delta"])
                for item in getattr(dataset, "index_map", [])
                if "time_delta" in item
            }
            if deltas:
                return sorted(deltas)

        return []

    def _get_stage1_max_time_delta(self, dataset=None):
        """Return the maximum Stage 1 time delta."""
        time_deltas = self._get_stage1_time_deltas(dataset=dataset)
        if time_deltas:
            return max(time_deltas)
        return float(self.config.get("ODE_MAX_T", 1.0))

    def _create_stage1_dataset(self):
        use_wmh = self.use_wmh_for_stage1_dataset
        if use_wmh is None:
            use_wmh = getattr(self, "use_wmh", False)
        max_slices = self.stage1_max_slices_per_patient
        if max_slices is None:
            max_slices = self.config["MAX_SLICES"]
        return FLAIREvolutionDataset(
            root_dir=self.config["ROOT_DIR"],
            max_slices_per_patient=max_slices,
            use_wmh=use_wmh,
            require_wmh_presence=self.require_wmh_presence,
            training_pairs=self.training_pairs,
            **self.stage1_dataset_kwargs,
        )

    def _load_folds_dict(self):
        fold_csv = self.config["FOLD_CSV"]
        if not os.path.exists(fold_csv):
            raise FileNotFoundError(f"Fold CSV not found at {fold_csv}")
        folds_dict = load_folds_from_csv(fold_csv)
        print(f"Loaded patient folds from {fold_csv}")
        return folds_dict

    def run(self):
        print("\n" + "=" * 60)
        print(f"Starting Experiment {self.experiment_number}: {self.run_title}")
        print("=" * 60 + "\n")

        predicted_flair_dir, ground_truth_wmh_dir = self._stage1_train_imageflownet()
        if predicted_flair_dir and ground_truth_wmh_dir:
            self._stage2_wmh_segmentation(predicted_flair_dir, ground_truth_wmh_dir)
        else:
            print("[Stage 2] Skipped because Stage 1 did not complete successfully.")
