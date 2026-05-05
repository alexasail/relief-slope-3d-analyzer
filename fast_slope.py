"""Fast Horn slope calculation (NumPy only)."""

from __future__ import annotations

import numpy as np


def calculate_slope_fast(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Calculate slope in degrees using Horn 3x3 kernel."""
    dem_f = np.asarray(dem, dtype=float)
    if dem_f.ndim != 2:
        raise ValueError("dem должен быть 2D-массивом.")

    dx = float(cell_size)
    if dx <= 0:
        raise ValueError("cell_size должен быть положительным числом.")
    dy = dx

    rows, cols = dem_f.shape
    slope = np.full((rows, cols), np.nan, dtype=float)
    if rows < 3 or cols < 3:
        return slope

    z1 = dem_f[:-2, :-2]
    z2 = dem_f[:-2, 1:-1]
    z3 = dem_f[:-2, 2:]
    z4 = dem_f[1:-1, :-2]
    z5 = dem_f[1:-1, 1:-1]
    z6 = dem_f[1:-1, 2:]
    z7 = dem_f[2:, :-2]
    z8 = dem_f[2:, 1:-1]
    z9 = dem_f[2:, 2:]

    valid = (
        np.isfinite(z1)
        & np.isfinite(z2)
        & np.isfinite(z3)
        & np.isfinite(z4)
        & np.isfinite(z5)
        & np.isfinite(z6)
        & np.isfinite(z7)
        & np.isfinite(z8)
        & np.isfinite(z9)
    )

    dzdx = ((z3 - z1) + 2.0 * (z6 - z4) + (z9 - z7)) / (8.0 * dx)
    dzdy = ((z7 - z1) + 2.0 * (z8 - z2) + (z9 - z3)) / (8.0 * dy)
    slope_inner = np.degrees(np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy)))
    slope[1:-1, 1:-1] = np.where(valid, slope_inner, np.nan)

    return slope

