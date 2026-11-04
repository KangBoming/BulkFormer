from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MissingnessCalibrationConfig:
    hidden_dim: int = 64
    reference_fraction: float = 0.15
    feature_scale: float = 0.15
    maximum_gain: float = 4.0
    maximum_absolute_offset: float = 2.0

    def __post_init__(self):
        values = (self.reference_fraction, self.feature_scale, self.maximum_gain,
                  self.maximum_absolute_offset)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("missingness calibration settings must be finite")
        if self.hidden_dim <= 0 or not 0 <= self.reference_fraction < 1:
            raise ValueError("invalid missingness calibration size or reference")
        if self.feature_scale <= 0 or self.maximum_gain <= 1 or self.maximum_absolute_offset <= 0:
            raise ValueError("invalid missingness calibration scale or bounds")


def input_missing_fraction(expression: Tensor, mask_value: float) -> Tensor:
    """Count unavailable inputs on the full gene axis; measured zeros are visible."""
    if expression.ndim != 2 or expression.shape[1] == 0:
        raise ValueError("expression must be a nonempty [batch, genes] matrix")
    return expression.eq(mask_value).float().mean(dim=1)


class MissingnessCalibration(nn.Module):
    """Learn a sample-wide output gain and offset from actual input missingness."""

    def __init__(self, config: MissingnessCalibrationConfig):
        super().__init__()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(1, config.hidden_dim), nn.SiLU(), nn.Linear(config.hidden_dim, 2),
        )
        # Every missing rate initially leaves the expression head's prediction intact.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, prediction: Tensor, missing_fraction: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if prediction.ndim != 2 or missing_fraction.shape != prediction.shape[:1]:
            raise ValueError("expected predictions [batch, genes] and missing fractions [batch]")
        # Keep the small calibrator and its bounded transforms in FP32 under BF16 training.
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            feature = (missing_fraction.float() - self.config.reference_fraction) / self.config.feature_scale
            values = self.network(feature.unsqueeze(-1))
            gain = (math.log(self.config.maximum_gain) * values[:, 0].tanh()).exp()
            offset = self.config.maximum_absolute_offset * values[:, 1].tanh()
            calibrated = prediction.float() * gain[:, None] + offset[:, None]
        return calibrated, gain, offset
