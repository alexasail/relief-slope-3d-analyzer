"""Benchmark Horn slope implementations."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve

from analyzer import horn_slope_degrees
from fast_slope import calculate_slope_fast


SIZES = [100, 500, 1000, 2000]
REPEATS = 5
CELL_SIZE = 30.0
OUTPUT_CSV = Path("benchmark_results.csv")


def generate_synthetic_dem(size: int, seed: int = 42) -> np.ndarray:
    """Generate synthetic DEM with random field and linear trend."""
    rng = np.random.default_rng(seed + size)
    base = rng.uniform(0.0, 500.0, size=(size, size))

    x = np.linspace(0.0, 1.0, size, dtype=float)
    y = np.linspace(0.0, 1.0, size, dtype=float)
    xx, yy = np.meshgrid(x, y)

    trend = 80.0 * xx + 55.0 * yy
    dem = base + trend
    return dem.astype(float)


def slope_scipy_reference(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Reference Horn slope via scipy.ndimage.convolve."""
    dem_f = np.asarray(dem, dtype=float)
    rows, cols = dem_f.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    if rows < 3 or cols < 3:
        return out

    dx = float(cell_size)
    dy = dx
    mask = np.isfinite(dem_f)
    dem_zeros = np.where(mask, dem_f, 0.0)

    kx = np.array(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        dtype=float,
    ) / (8.0 * dx)
    ky = np.array(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        dtype=float,
    ) / (8.0 * dy)

    dzdx = convolve(dem_zeros, kx, mode="constant", cval=0.0)
    dzdy = convolve(dem_zeros, ky, mode="constant", cval=0.0)

    valid_count = convolve(
        mask.astype(np.uint8),
        np.ones((3, 3), dtype=np.uint8),
        mode="constant",
        cval=0,
    )
    valid_window = valid_count == 9
    valid_window[[0, -1], :] = False
    valid_window[:, [0, -1]] = False

    slope = np.degrees(np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy)))
    out[valid_window] = slope[valid_window]
    return out


def calculate_slope_naive(dem: np.ndarray, cell_size: float) -> np.ndarray:
    """Naive Horn implementation with nested loops."""
    dem_f = np.asarray(dem, dtype=float)
    rows, cols = dem_f.shape
    slope = np.full((rows, cols), np.nan, dtype=float)
    if rows < 3 or cols < 3:
        return slope

    cs = float(cell_size)
    if cs <= 0:
        raise ValueError("cell_size должен быть положительным числом.")

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            window = dem_f[i - 1 : i + 2, j - 1 : j + 2]
            if np.any(np.isnan(window)):
                continue

            z1, z2, z3 = window[0, 0], window[0, 1], window[0, 2]
            z4, z5, z6 = window[1, 0], window[1, 1], window[1, 2]
            z7, z8, z9 = window[2, 0], window[2, 1], window[2, 2]

            dzdx = ((z3 - z1) + 2.0 * (z6 - z4) + (z9 - z7)) / (8.0 * cs)
            dzdy = ((z7 - z1) + 2.0 * (z8 - z2) + (z9 - z3)) / (8.0 * cs)
            slope[i, j] = np.degrees(np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy)))
    return slope


def mean_runtime(func, dem: np.ndarray, repeats: int = REPEATS) -> tuple[float, np.ndarray]:
    """Return mean runtime and result of the last run."""
    timings: list[float] = []
    last_result: np.ndarray | None = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        last_result = func(dem, CELL_SIZE)
        t1 = time.perf_counter()
        timings.append(t1 - t0)
    return float(np.mean(timings)), last_result if last_result is not None else np.array([])


def print_markdown_table(rows: list[dict[str, float]]) -> None:
    print("\n## Benchmark Table (Markdown)\n")
    print("| DEM size | Naive, s | Legacy, s | Fast, s | Scipy, s | Fast vs Naive | Fast vs Legacy | Fast vs Scipy |")
    print("|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {int(r['size'])}x{int(r['size'])} | "
            f"{r['naive_s']:.6f} | {r['legacy_s']:.6f} | {r['fast_s']:.6f} | {r['scipy_s']:.6f} | "
            f"{r['speedup_fast_vs_naive']:.2f}x | "
            f"{r['speedup_fast_vs_legacy']:.2f}x | {r['speedup_fast_vs_scipy']:.2f}x |"
        )


def save_csv(rows: list[dict[str, float]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "size",
                "naive_s",
                "legacy_s",
                "fast_s",
                "scipy_s",
                "speedup_fast_vs_naive",
                "speedup_fast_vs_legacy",
                "speedup_fast_vs_scipy",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def build_plots(rows: list[dict[str, float]], error_map_500: np.ndarray) -> None:
    sizes = [int(r["size"]) for r in rows]
    naive_t = [r["naive_s"] for r in rows]
    legacy_t = [r["legacy_s"] for r in rows]
    fast_t = [r["fast_s"] for r in rows]
    scipy_t = [r["scipy_s"] for r in rows]
    speedup_naive = [r["speedup_fast_vs_naive"] for r in rows]
    speedup_legacy = [r["speedup_fast_vs_legacy"] for r in rows]
    speedup_scipy = [r["speedup_fast_vs_scipy"] for r in rows]

    fig1 = plt.figure(figsize=(8, 5))
    plt.plot(sizes, naive_t, marker="o", label="naive")
    plt.plot(sizes, legacy_t, marker="o", label="legacy")
    plt.plot(sizes, fast_t, marker="o", label="fast")
    plt.plot(sizes, scipy_t, marker="o", label="scipy")
    plt.yscale("log")
    plt.xlabel("DEM size (N x N)")
    plt.ylabel("Runtime, seconds (log scale)")
    plt.title("Runtime vs DEM size")
    plt.grid(alpha=0.25)
    plt.legend()
    fig1.tight_layout()
    fig1.savefig("plot_time_vs_size.png", dpi=150)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(9, 5))
    x = np.arange(len(sizes))
    width = 0.25
    plt.bar(x - width, speedup_naive, width=width, label="fast / naive")
    plt.bar(x, speedup_legacy, width=width, label="fast / legacy")
    plt.bar(x + width, speedup_scipy, width=width, label="fast / scipy")
    plt.xticks(x, [f"{s}x{s}" for s in sizes])
    plt.ylabel("Speedup (x)")
    plt.title("Fast method speedup")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    fig2.tight_layout()
    fig2.savefig("plot_speedup.png", dpi=150)
    plt.close(fig2)

    fig3 = plt.figure(figsize=(7, 6))
    im = plt.imshow(error_map_500, cmap="magma", origin="upper")
    plt.title("FAST - SCIPY absolute error map (500x500)")
    plt.colorbar(im, label="Absolute error, degrees")
    fig3.tight_layout()
    fig3.savefig("plot_error_map.png", dpi=150)
    plt.close(fig3)


def main() -> None:
    rows: list[dict[str, float]] = []
    error_map_500: np.ndarray | None = None
    mae_500 = float("nan")
    max_err_500 = float("nan")

    for size in SIZES:
        dem = generate_synthetic_dem(size)

        naive_time, _ = mean_runtime(calculate_slope_naive, dem)
        legacy_time, legacy_slope = mean_runtime(horn_slope_degrees, dem)
        fast_time, fast_slope = mean_runtime(calculate_slope_fast, dem)
        scipy_time, scipy_slope = mean_runtime(slope_scipy_reference, dem)

        speedup_fast_vs_naive = naive_time / fast_time if fast_time > 0 else np.nan
        speedup_fast_vs_legacy = legacy_time / fast_time if fast_time > 0 else np.nan
        speedup_fast_vs_scipy = scipy_time / fast_time if fast_time > 0 else np.nan

        rows.append(
            {
                "size": float(size),
                "naive_s": naive_time,
                "legacy_s": legacy_time,
                "fast_s": fast_time,
                "scipy_s": scipy_time,
                "speedup_fast_vs_naive": float(speedup_fast_vs_naive),
                "speedup_fast_vs_legacy": float(speedup_fast_vs_legacy),
                "speedup_fast_vs_scipy": float(speedup_fast_vs_scipy),
            }
        )

        if size == 500:
            valid = np.isfinite(fast_slope) & np.isfinite(scipy_slope)
            valid[[0, -1], :] = False
            valid[:, [0, -1]] = False
            err = np.abs(fast_slope - scipy_slope)
            err_valid = err[valid]
            mae_500 = float(np.mean(err_valid))
            max_err_500 = float(np.max(err_valid))
            error_map_500 = err

    save_csv(rows, OUTPUT_CSV)
    if error_map_500 is None:
        raise RuntimeError("Не удалось сформировать error map для DEM 500x500.")
    build_plots(rows, error_map_500)

    print(f"FAST vs SCIPY: MAE = {mae_500:.6f}°, MAX = {max_err_500:.6f}°")
    print_markdown_table(rows)
    print(f"\nCSV: {OUTPUT_CSV.resolve()}")
    print(f"Plots: {Path('plot_time_vs_size.png').resolve()}, {Path('plot_speedup.png').resolve()}, {Path('plot_error_map.png').resolve()}")


if __name__ == "__main__":
    main()

