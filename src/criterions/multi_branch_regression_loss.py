"""Multi-branch (per-vital) regression loss for the MD-ViSCo refinement model.

Implements the paper's headline BP-refinement MAE objective (§III.D, Eq. for
L_MAE): with both ECG and PPG present per sample, each single-source branch is
supervised *independently* against the same SBP/DBP target and the per-branch
MAEs are SUMMED over i in {ECG, PPG}:

    L_MAE = lambda * sum_{i in {ECG, PPG}} ( |sbp_i - SBP| + |dbp_i - DBP| )

This differs from the stock regression term, which computes MAE on BPModel's
*aggregated* (averaged) prediction — that aggregation matches the paper's
Appendix-E multi-input ablation, not the headline (and the paper reports that
average-then-single-modality inference degrades). Supervising each branch
independently is what makes single-modality inference match training.

The per-vital predictions are read from the model output `per_vital_bp` dict
(keys `<vital>_sbp` / `<vital>_dbp`), forwarded as kwargs by the combined
criterion (RegressionMultiWeightedContrastiveLoss passes its kwargs through to
the regression term). When no per-vital outputs are present (e.g. a non-BPModel
path), it falls back to MAE on `input` so the term is safe to drop into any
regression pipeline.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812  # conventional alias F for functional
from hydra.core.config_store import ConfigStore

from src.criterions.base_criterion import BaseCriterion
from src.criterions.base_criterion import CriterionBaseConfig
from src.criterions.base_criterion import ReductionType


@dataclass
class MultiBranchL1LossConfig(CriterionBaseConfig):
    """Configuration for the multi-branch (per-vital) L1 regression loss.

    Attributes:
        scale_factor: Multiplier applied to the summed per-branch MAE. Defaults
            to 1.0 (same magnitude as the stock l1_loss regression term). Set to
            the paper's lambda_MAE = 0.001 to reproduce the headline loss balance
            against the WCL terms.
        per_vital_key: Key in the criterion kwargs holding the per-vital BP dict
            ({'<vital>_sbp': [B,1], '<vital>_dbp': [B,1], ...}). Matches BPModel's
            `per_vital_bp` output.
    """

    _target_: str = "src.criterions.multi_branch_regression_loss.MultiBranchL1Loss"
    name: str = "multi_branch_l1_loss"
    reduction: ReductionType = ReductionType.MEAN
    scale_factor: float = 1.0
    per_vital_key: str = "per_vital_bp"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        if not isinstance(self.reduction, ReductionType):
            raise ValueError(
                f"reduction must be a ReductionType enum, got {type(self.reduction)}"
            )
        if self.scale_factor < 0:
            raise ValueError(
                f"scale_factor must be non-negative, got {self.scale_factor}"
            )


class MultiBranchL1Loss(BaseCriterion):
    """Per-vital summed L1 (MAE) regression loss for BPModel refinement.

    See module docstring. The returned (unreduced) tensor is the per-branch sum
    of element-wise MAE [B, num_targets]; BaseCriterion then applies the
    configured reduction (mean by default), matching the stock l1_loss term.
    """

    def __init__(
        self,
        reduction: ReductionType = ReductionType.MEAN,
        device: torch.device | None = None,
        name: str = "multi_branch_l1_loss",
        log_loss: bool = False,
        scale_factor: float = 1.0,
        per_vital_key: str = "per_vital_bp",
        **kwargs,
    ):
        """Initialize the multi-branch L1 loss.

        Args:
            reduction: Reduction method for the loss.
            device: Device to compute the loss on.
            name: Name for logging purposes.
            log_loss: Whether to log loss values.
            scale_factor: Multiplier applied to the summed per-branch MAE.
            per_vital_key: kwargs key holding the per-vital BP dict.
            **kwargs: Extra config fields (e.g. ``enabled``) forwarded to
                BaseCriterion.
        """
        super().__init__(
            reduction=reduction, device=device, name=name, log_loss=log_loss, **kwargs
        )
        self.scale_factor = scale_factor
        self.per_vital_key = per_vital_key

    @staticmethod
    def _as_bx1(value: torch.Tensor) -> torch.Tensor:
        """Coerce a per-vital scalar prediction [B] or [B, 1] to shape [B, 1].

        BPModel emits [B, 1] per vital; a bare [B] is unsqueezed. Any wider
        tensor is left untouched (and will fail loudly at the cat below) rather
        than silently truncated.
        """
        return value.unsqueeze(-1) if value.dim() == 1 else value

    def _collect_branch_predictions(
        self, per_vital: dict[str, torch.Tensor]
    ) -> list[torch.Tensor]:
        """Assemble per-vital [B, 2] (SBP, DBP) prediction tensors.

        Args:
            per_vital: Dict with keys '<vital>_sbp' / '<vital>_dbp'.

        Returns:
            List of [B, 2] tensors, one per vital that has BOTH sbp and dbp.
        """
        # Order is irrelevant — the branch losses are summed downstream.
        vitals = [key[: -len("_sbp")] for key in per_vital if key.endswith("_sbp")]
        branches: list[torch.Tensor] = []
        for vital in vitals:
            sbp = per_vital.get(f"{vital}_sbp")
            dbp = per_vital.get(f"{vital}_dbp")
            if isinstance(sbp, torch.Tensor) and isinstance(dbp, torch.Tensor):
                branches.append(
                    torch.cat([self._as_bx1(sbp), self._as_bx1(dbp)], dim=1)
                )
        return branches

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, **kwargs
    ) -> torch.Tensor:
        """Compute the summed per-branch L1 loss.

        Args:
            input: Aggregated BP predictions [B, 2] (used only as fallback).
            target: Ground-truth normalized BP [B, 2] (SBP, DBP).
            **kwargs: Criterion context; must contain ``per_vital_key`` (the
                per-vital BP dict) for per-branch supervision.

        Returns:
            Unreduced loss tensor [B, 2] (per-branch sum), scaled by
            ``scale_factor``. BaseCriterion applies the final reduction.
        """
        per_vital = kwargs.get(self.per_vital_key)
        branches: list[torch.Tensor] = []
        if isinstance(per_vital, dict) and per_vital:
            branches = self._collect_branch_predictions(per_vital)

        if not branches:
            # No per-vital outputs available -> supervise the aggregated prediction.
            return self.scale_factor * F.l1_loss(input, target, reduction="none")

        # Sum the per-branch [B, 2] MAEs (paper: L_MAE summed over i in {ECG, PPG}).
        branch_losses = torch.stack(
            [F.l1_loss(pred, target, reduction="none") for pred in branches]
        )  # [num_vitals, B, 2]
        return self.scale_factor * branch_losses.sum(dim=0)


cs = ConfigStore.instance()
cs.store(
    group="criterion", name="base_multi_branch_l1_loss", node=MultiBranchL1LossConfig
)
