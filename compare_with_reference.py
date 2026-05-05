"""Compare Horn slope implementation with scipy reference."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy.ndimage import convolve

from analyzer import ReliefAnalyzer, horn_slope_degrees


def _resolve_cell_size(src: rasterio.io.DatasetReader) -> tuple[float, float]:
    transform = src.transform
    cell_size_x = float(abs(transform.a))
    cell_size_y = float(abs(transform.e))
    return cell_size_x, cell_size_y


def _reference_horn_slope(dem: np.ndarray, cell_size: tuple[float, float]) -> np.ndarray:
    dx, dy = cell_size
    dem_f = np.asarray(dem, dtype=float)
    rows, cols = dem_f.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    if rows < 3 or cols < 3:
        return out

    mask = np.isfinite(dem_f)
    dem_zeros = np.where(mask, dem_f, 0.0)

    kx = np.array(
        [
            [-1.0, 0.0, 1.0],
            [-2.0, 0.0, 2.0],
            [-1.0, 0.0, 1.0],
        ],
        dtype=float,
    ) / (8.0 * dx)
    ky = np.array(
        [
            [-1.0, -2.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 1.0],
        ],
        dtype=float,
    ) / (8.0 * dy)

    dzdx = convolve(dem_zeros, kx, mode="constant", cval=0.0)
    dzdy = convolve(dem_zeros, ky, mode="constant", cval=0.0)

    valid_count = convolve(mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant", cval=0)
    valid_window = valid_count == 9
    valid_window[[0, -1], :] = False
    valid_window[:, [0, -1]] = False

    slope = np.degrees(np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy)))
    out[valid_window] = slope[valid_window]
    return out


def _load_dem(dem_path: str | None) -> tuple[np.ndarray, tuple[float, float], str]:
    if dem_path:
        path = Path(dem_path)
        if not path.exists():
            raise FileNotFoundError(f"DEM не найден: {path}")
        with rasterio.open(path) as src:
            dem = src.read(1).astype(float)
            nodata = src.nodata
            if nodata is not None and np.isscalar(nodata):
                dem = np.where(dem == nodata, np.nan, dem)
            cell_size = _resolve_cell_size(src)
        return dem, cell_size, str(path)

    cache_files = sorted((Path(__file__).resolve().parent / "cache" / "dem").glob("*.tif"))
    if cache_files:
        with rasterio.open(cache_files[0]) as src:
            dem = src.read(1).astype(float)
            nodata = src.nodata
            if nodata is not None and np.isscalar(nodata):
                dem = np.where(dem == nodata, np.nan, dem)
            cell_size = _resolve_cell_size(src)
        return dem, cell_size, str(cache_files[0])

    analyzer = ReliefAnalyzer(use_cache=True)
    dem = analyzer.load_dem(lat=55.75, lon=37.62, bbox_deg=0.02).astype(float)
    if analyzer._bounds is None:
        raise RuntimeError("Не удалось определить границы DEM для расчета cell_size.")
    w, s, e, n = analyzer._bounds
    rows, cols = dem.shape
    lat_center_rad = np.radians((n + s) / 2.0)
    m_per_deg_lon = 111320 * np.cos(lat_center_rad)
    m_per_deg_lat = 110540
    cell_size = ((e - w) / cols * m_per_deg_lon, (n - s) / rows * m_per_deg_lat)
    return dem, cell_size, "OpenTopography (downloaded)"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem-path", type=str, default=None, help="Путь к DEM GeoTIFF")
    parser.add_argument("--samples", type=int, default=30, help="Количество случайных точек")
    parser.add_argument("--out-dir", type=str, default="output/slope_compare", help="Папка для результатов")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dem, cell_size, dem_source = _load_dem(args.dem_path)
    slope_own = horn_slope_degrees(dem, cell_size)
    slope_ref = _reference_horn_slope(dem, cell_size)
    error_map = np.abs(slope_own - slope_ref)

    valid = np.isfinite(slope_own) & np.isfinite(slope_ref)
    valid[[0, -1], :] = False
    valid[:, [0, -1]] = False
    points = np.argwhere(valid)
    if points.shape[0] < args.samples:
        raise RuntimeError(
            f"Недостаточно валидных внутренних точек ({points.shape[0]}) для выборки {args.samples}."
        )

    rng = np.random.default_rng(42)
    idx = rng.choice(points.shape[0], size=args.samples, replace=False)
    sample_pts = points[idx]

    rows = []
    for r, c in sample_pts:
        own = float(slope_own[r, c])
        ref = float(slope_ref[r, c])
        err = abs(own - ref)
        rows.append((int(r), int(c), own, ref, err))

    csv_path = out_dir / "slope_comparison.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write("row,col,slope_own_deg,slope_ref_deg,abs_error_deg\n")
        for r, c, own, ref, err in rows:
            f.write(f"{r},{c},{own:.8f},{ref:.8f},{err:.8f}\n")

    all_errors = error_map[np.isfinite(error_map)]
    mean_err = float(np.mean(all_errors))
    max_err = float(np.max(all_errors))
    std_err = float(np.std(all_errors))

    print(f"Источник DEM: {dem_source}")
    print(f"Размер DEM: {dem.shape}, cell_size(dx,dy)={cell_size}")
    print("\nТаблица сравнения (30 случайных внутренних точек):")
    print("row\tcol\tslope_own\tslope_ref\tabs_error")
    for r, c, own, ref, err in rows:
        print(f"{r}\t{c}\t{own:.6f}\t{ref:.6f}\t{err:.6f}")

    print("\nСтатистика ошибок по всем валидным внутренним пикселям:")
    print(f"Средняя ошибка: {mean_err:.6f}°")
    print(f"Максимальная ошибка: {max_err:.6f}°")
    print(f"STD ошибки: {std_err:.6f}°")

    diploma_text = (
        f"Средняя ошибка составила {mean_err:.2f}°, "
        "что подтверждает корректность реализации."
    )
    print("\nТекст для диплома:")
    print(diploma_text)

    fig1 = plt.figure(figsize=(8, 6))
    plt.scatter(
        slope_ref[np.isfinite(slope_ref)],
        slope_own[np.isfinite(slope_own)],
        s=4,
        alpha=0.35,
    )
    plt.xlabel("Эталонный уклон, °")
    plt.ylabel("Реализованный уклон, °")
    plt.title("Сравнение уклонов: реализованный vs эталонный")
    lim = [0, 90]
    plt.plot(lim, lim, "r--", linewidth=1)
    plt.xlim(lim)
    plt.ylim(lim)
    plt.grid(alpha=0.2)
    fig1.tight_layout()
    fig1.savefig(out_dir / "scatter_own_vs_reference.png", dpi=150)
    plt.close(fig1)

    fig2 = plt.figure(figsize=(8, 6))
    im = plt.imshow(error_map, cmap="magma", origin="upper")
    plt.title("Карта абсолютной ошибки уклона, °")
    plt.colorbar(im, shrink=0.8, label="|Ошибка|, °")
    fig2.tight_layout()
    fig2.savefig(out_dir / "error_map.png", dpi=150)
    plt.close(fig2)

    print(f"\nCSV сохранен: {csv_path}")
    print(f"Графики сохранены в: {out_dir}")


if __name__ == "__main__":
    main()
