# -*- coding: utf-8 -*-
"""
Explicit experiment matrix for Flair->Flair variants:
- BL
- L1
- ODEY-BL
- ODEM-BL
- ODEY-L1
- ODEM-L1

All classes intentionally reuse the current ExperimentBL pipeline so behavior stays
aligned with flair_to_flair.py and the shared utils.py train/val logic.
"""

import torch.nn as nn

from flair_to_flair import ExperimentBL
from losses import L1SSIMLoss


class ImageFlowNetWithL1SSIMLoss(nn.Module):
    """Keep original ImageFlowNet MSE and add an L1+SSIM term."""

    def __init__(self, alpha=0.7, l1_ssim_weight=1.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.l1_ssim = L1SSIMLoss(alpha=alpha)
        self.l1_ssim_weight = l1_ssim_weight

    def forward(self, pred, target):
        return self.mse(pred, target) + self.l1_ssim_weight * self.l1_ssim(pred, target)


class ExperimentBLAllPairs(ExperimentBL):
    """BL: all possible time points with original ImageFlowNet loss."""

    run_title = "BL: FLAIR -> FLAIR (all pairs, original ImageFlowNet loss)"


class ExperimentL1AllPairs(ExperimentBL):
    """L1: all possible time points with original loss + L1+SSIM novelty."""

    run_title = "L1: FLAIR -> FLAIR (all pairs, ImageFlowNet + L1SSIM)"

    def _create_recon_loss(self):
        return ImageFlowNetWithL1SSIMLoss(
            alpha=self.config.get("L1_SSIM_ALPHA", 0.7),
            l1_ssim_weight=self.config.get("L1_SSIM_WEIGHT", 1.0),
        )


class ExperimentODEYBL(ExperimentBLAllPairs):
    """ODEY-BL: year-scaled ODE time for BL."""

    ode_max_t_override = 9.0
    run_title = "ODEY-BL: FLAIR -> FLAIR (years, original ImageFlowNet loss)"


class ExperimentODEMBL(ExperimentBLAllPairs):
    """ODEM-BL: month-scaled ODE time for BL."""

    ode_max_t_override = 108.0
    run_title = "ODEM-BL: FLAIR -> FLAIR (months, original ImageFlowNet loss)"


class ExperimentODEYL1(ExperimentL1AllPairs):
    """ODEY-L1: year-scaled ODE time for L1+SSIM variant."""

    ode_max_t_override = 9.0
    run_title = "ODEY-L1: FLAIR -> FLAIR (years, ImageFlowNet + L1SSIM)"


class ExperimentODEML1(ExperimentL1AllPairs):
    """ODEM-L1: month-scaled ODE time for L1+SSIM variant."""

    ode_max_t_override = 108.0
    run_title = "ODEM-L1: FLAIR -> FLAIR (months, ImageFlowNet + L1SSIM)"
