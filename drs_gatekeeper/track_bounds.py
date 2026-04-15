from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TrackBoundsEstimate:
    half_width_mean: float
    half_width_std: float
    center_offset_mean: float = 0.0
    center_offset_std: float = 0.0

    @classmethod
    def unknown(cls) -> "TrackBoundsEstimate":
        return cls(
            half_width_mean=np.nan,
            half_width_std=np.nan,
            center_offset_mean=0.0,
            center_offset_std=0.0,
        )

    @classmethod
    def from_track_width(
        cls,
        half_width: float,
        relative_uncertainty: float = 0.0,
        center_offset_mean: float = 0.0,
        center_offset_std: float = 0.0,
    ) -> "TrackBoundsEstimate":
        half_width = max(float(half_width), 1e-3)
        return cls(
            half_width_mean=half_width,
            half_width_std=abs(float(relative_uncertainty)) * half_width,
            center_offset_mean=float(center_offset_mean),
            center_offset_std=max(float(center_offset_std), 0.0),
        )

    def sample_env_params(self, M: int, N: int, T: int, rng) -> np.ndarray:
        if not np.isfinite(self.half_width_mean):
            return np.zeros((int(M), int(N), int(T), 2), dtype=np.float32)

        half_width = np.full((int(M), int(N), 1), self.half_width_mean, dtype=np.float32)
        center_offset = np.full((int(M), int(N), 1), self.center_offset_mean, dtype=np.float32)

        if self.half_width_std > 0.0:
            half_width += rng.normal(
                loc=0.0,
                scale=self.half_width_std,
                size=half_width.shape,
            ).astype(np.float32)
        if self.center_offset_std > 0.0:
            center_offset += rng.normal(
                loc=0.0,
                scale=self.center_offset_std,
                size=center_offset.shape,
            ).astype(np.float32)

        half_width = np.maximum(half_width, 1e-3)
        env = np.stack([half_width[..., 0], center_offset[..., 0]], axis=-1)
        env = np.repeat(env[:, :, None, :], int(T), axis=2)
        return env.astype(np.float32, copy=False)
