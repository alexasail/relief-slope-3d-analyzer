"""
ReliefAnalyzer — анализ цифровых моделей рельефа (DEM).
Загрузка DEM из OpenTopography, медианная фильтрация, уклон по методу Хорна (3×3),
визуализация в Matplotlib (карты высот и уклонов), сохранение в PNG.
"""

import hashlib
import io
import logging
import os
import struct
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
from scipy.ndimage import gaussian_filter, median_filter, zoom

logger = logging.getLogger(__name__)

# Базовый URL API OpenTopography (Global DEM)
OPENTOPOGRAPHY_API_BASE = "https://portal.opentopography.org/API/globaldem"
# Тип DEM по умолчанию: SRTM GL1 30m (хороший баланс качества и лимитов)
DEFAULT_DEM_TYPE = "SRTMGL1"
# Набор источников для автоматического fallback, если у первого нет покрытия.
DEM_FALLBACK_TYPES = ("SRTMGL1", "COP30", "ASTER")
# Максимальная сторона bbox в градусах (ограничение API ~450000 km² для 30m)
MAX_BBOX_DEG = 0.5
# Размер медианного фильтра для подавления шума
MEDIAN_FILTER_SIZE = 3
# Локальный кэш DEM
CACHE_DIR = Path(__file__).resolve().parent / "cache" / "dem"


def _cache_path(south: float, north: float, west: float, east: float, dem_type: str) -> Path:
    key = f"{south:.6f}_{north:.6f}_{west:.6f}_{east:.6f}_{dem_type}"
    name = hashlib.sha256(key.encode()).hexdigest()[:16] + ".tif"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / name


def _fetch_dem_opentopography(
    south: float,
    north: float,
    west: float,
    east: float,
    dem_type: str = DEFAULT_DEM_TYPE,
    api_key: Optional[str] = None,
    use_cache: bool = True,
) -> Tuple[np.ndarray, float, float, float, float]:
    """
    Загружает DEM из OpenTopography API (GeoTIFF).
    Возвращает: (elevation_array, west, south, east, north) в градусах и метрах.
    """
    api_key = api_key or os.getenv("OPENTOPOGRAPHY_API_KEY") or "demoapikeyot2022"
    cache_file = _cache_path(south, north, west, east, dem_type)

    if use_cache and cache_file.exists():
        try:
            import rasterio
            with rasterio.open(cache_file) as src:
                dem = src.read(1)
                bounds = src.bounds
                return dem, float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)
        except Exception as e:
            logger.warning("Не удалось прочитать кэш DEM: %s", e)

    params = {
        "demtype": dem_type,
        "south": south,
        "north": north,
        "west": west,
        "east": east,
        "outputFormat": "GTiff",
        "API_Key": api_key,
    }
    resp = requests.get(OPENTOPOGRAPHY_API_BASE, params=params, timeout=120)
    resp.raise_for_status()
    content = resp.content
    if len(content) < 1000:
        # При отсутствии покрытия API может вернуть краткий текст/JSON вместо GeoTIFF.
        # В этом случае вызывающий код попробует другой dem_type.
        text = content.decode("utf-8", errors="ignore").strip()
        raise ValueError(
            f"OpenTopography вернул невалидный DEM для {dem_type}. "
            f"Возможная причина: нет покрытия в этой области. Ответ API: {text[:200]}"
        )

    try:
        import rasterio
    except ImportError:
        raise ImportError("Установите rasterio: pip install rasterio")

    with rasterio.open(io.BytesIO(content)) as src:
        dem = src.read(1)
        bounds = src.bounds
        # no-data обычно отрицательные большие значения или заданы в meta
        nodata = getattr(src, "nodata", -32768)
        if nodata is not None and np.isscalar(nodata):
            dem = np.where(dem == nodata, np.nan, dem)
        # Сохраняем в кэш
        if use_cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            with rasterio.open(cache_file, "w", **src.meta) as dst:
                dst.write(dem, 1)

    return dem, float(bounds.left), float(bounds.bottom), float(bounds.right), float(bounds.top)


def horn_slope_degrees(
    dem: np.ndarray,
    cell_size: Union[float, Tuple[float, float]],
) -> np.ndarray:
    """
    Уклон в градусах по канонической формуле Хорна (3x3 окно), полностью векторизованно.
    cell_size: float (квадратные ячейки) или tuple(dx, dy) в метрах.

    Правила:
    - Граничные пиксели всегда np.nan.
    - Если в окне 3x3 есть хотя бы один NaN, центр окна = np.nan.
    """
    dem_f = np.asarray(dem, dtype=float)
    if dem_f.ndim != 2:
        raise ValueError("dem должен быть 2D-массивом.")

    if np.isscalar(cell_size):
        cell_size_x_m = float(cell_size)
        cell_size_y_m = float(cell_size)
    else:
        if len(cell_size) != 2:
            raise ValueError("cell_size должен быть числом или кортежем (dx, dy).")
        cell_size_x_m = float(cell_size[0])
        cell_size_y_m = float(cell_size[1])
    if cell_size_x_m <= 0 or cell_size_y_m <= 0:
        raise ValueError("Размер ячейки должен быть положительным.")

    rows, cols = dem_f.shape
    out = np.full((rows, cols), np.nan, dtype=float)
    if rows < 3 or cols < 3:
        return out

    # z1..z9 для окна 3x3:
    # [z1 z2 z3]
    # [z4 z5 z6]
    # [z7 z8 z9]
    z1 = dem_f[:-2, :-2]
    z2 = dem_f[:-2, 1:-1]
    z3 = dem_f[:-2, 2:]
    z4 = dem_f[1:-1, :-2]
    z6 = dem_f[1:-1, 2:]
    z7 = dem_f[2:, :-2]
    z8 = dem_f[2:, 1:-1]
    z9 = dem_f[2:, 2:]

    window_has_nan = (
        ~np.isfinite(z1) | ~np.isfinite(z2) | ~np.isfinite(z3) |
        ~np.isfinite(z4) | ~np.isfinite(z6) |
        ~np.isfinite(z7) | ~np.isfinite(z8) | ~np.isfinite(z9) |
        ~np.isfinite(dem_f[1:-1, 1:-1])
    )

    dzdx = ((z3 - z1) + 2.0 * (z6 - z4) + (z9 - z7)) / (8.0 * cell_size_x_m)
    dzdy = ((z7 - z1) + 2.0 * (z8 - z2) + (z9 - z3)) / (8.0 * cell_size_y_m)
    slope_deg = np.degrees(np.arctan(np.sqrt(dzdx * dzdx + dzdy * dzdy)))
    slope_deg = np.where(window_has_nan, np.nan, slope_deg)

    out[1:-1, 1:-1] = slope_deg
    return out


def validate_slope(slope: np.ndarray, dem: np.ndarray) -> list[str]:
    """
    Проверяет корректность карты уклонов.
    Возвращает список ошибок; если ошибок нет, печатает "Проверка пройдена".
    """
    errors: list[str] = []
    slope_arr = np.asarray(slope, dtype=float)
    dem_arr = np.asarray(dem, dtype=float)

    if slope_arr.shape != dem_arr.shape:
        errors.append(
            f"Размеры не совпадают: slope={slope_arr.shape}, dem={dem_arr.shape}."
        )

    invalid_values = np.isfinite(slope_arr) & ((slope_arr < 0.0) | (slope_arr > 90.0))
    if np.any(invalid_values):
        errors.append("Найдены значения уклона вне диапазона 0-90 градусов.")

    if slope_arr.ndim == 2 and slope_arr.shape[0] >= 1 and slope_arr.shape[1] >= 1:
        borders = [
            slope_arr[0, :],
            slope_arr[-1, :],
            slope_arr[:, 0],
            slope_arr[:, -1],
        ]
        for border in borders:
            if not np.all(np.isnan(border)):
                errors.append("Граничные строки и столбцы должны состоять только из NaN.")
                break

    if not errors:
        print("Проверка пройдена")
    return errors


class ReliefAnalyzer:
    """
    Анализатор рельефа: загрузка DEM (OpenTopography), медианная фильтрация,
    расчёт уклона по методу Хорна, визуализация — карты высот и уклонов в PNG.
    """

    def __init__(
        self,
        dem_type: str = DEFAULT_DEM_TYPE,
        api_key: Optional[str] = None,
        cache_dir: Optional[os.PathLike] = None,
        use_cache: bool = True,
    ):
        self.dem_type = dem_type
        self.api_key = api_key or os.getenv("OPENTOPOGRAPHY_API_KEY")
        self.use_cache = use_cache
        if cache_dir is not None:
            global CACHE_DIR
            CACHE_DIR = Path(cache_dir)

        self._elevation: Optional[np.ndarray] = None
        self._slope: Optional[np.ndarray] = None
        self._bounds: Optional[Tuple[float, float, float, float]] = None  # west, south, east, north

    def load_dem(
        self,
        lat: float,
        lon: float,
        bbox_deg: Optional[float] = None,
    ) -> np.ndarray:
        """
        Загружает DEM для области с центром (lat, lon).
        bbox_deg — полуразмер стороны в градусах (по умолчанию ограничиваем MAX_BBOX_DEG).
        """
        if bbox_deg is None:
            bbox_deg = min(0.05, MAX_BBOX_DEG)
        bbox_deg = min(bbox_deg, MAX_BBOX_DEG)
        south = lat - bbox_deg
        north = lat + bbox_deg
        west = lon - bbox_deg
        east = lon + bbox_deg
        dem = None
        w = s = e = n = 0.0
        errors: list[str] = []
        dem_candidates = [self.dem_type] + [
            d for d in DEM_FALLBACK_TYPES if d != self.dem_type
        ]
        for dem_type in dem_candidates:
            try:
                dem, w, s, e, n = _fetch_dem_opentopography(
                    south,
                    north,
                    west,
                    east,
                    dem_type=dem_type,
                    api_key=self.api_key,
                    use_cache=self.use_cache,
                )
                if dem is not None and np.isfinite(dem).any():
                    self.dem_type = dem_type
                    break
                errors.append(f"{dem_type}: DEM без валидных значений")
            except Exception as exc:
                errors.append(f"{dem_type}: {exc}")

        if dem is None or not np.isfinite(dem).any():
            raise ValueError(
                "Не удалось загрузить DEM для этой области. "
                "Пробованы источники: "
                + ", ".join(dem_candidates)
                + ". "
                + " | ".join(errors[:3])
            )

        self._bounds = (w, s, e, n)
        # Медианная фильтрация для устранения шума
        dem_smooth = median_filter(np.nan_to_num(dem, nan=0.0), size=MEDIAN_FILTER_SIZE)
        nan_mask = ~np.isfinite(dem)
        if np.any(nan_mask):
            dem_smooth[nan_mask] = np.nan
        self._elevation = dem_smooth
        return self._elevation

    def load_dem_by_corners(
        self,
        left_top_lat: float,
        left_top_lon: float,
        right_bottom_lat: float,
        right_bottom_lon: float,
    ) -> np.ndarray:
        """
        Загружает DEM по двум углам области: левый верхний и правый нижний.
        """
        north = max(left_top_lat, right_bottom_lat)
        south = min(left_top_lat, right_bottom_lat)
        west = min(left_top_lon, right_bottom_lon)
        east = max(left_top_lon, right_bottom_lon)

        if north <= south or east <= west:
            raise ValueError("Некорректные углы области: проверьте порядок и значения координат.")

        max_side_deg = MAX_BBOX_DEG * 2.0
        if (north - south) > max_side_deg or (east - west) > max_side_deg:
            raise ValueError(
                f"Слишком большая область. Максимум: {max_side_deg:.2f}° по широте и долготе."
            )

        dem = None
        w = s = e = n = 0.0
        errors: list[str] = []
        dem_candidates = [self.dem_type] + [
            d for d in DEM_FALLBACK_TYPES if d != self.dem_type
        ]
        for dem_type in dem_candidates:
            try:
                dem, w, s, e, n = _fetch_dem_opentopography(
                    south,
                    north,
                    west,
                    east,
                    dem_type=dem_type,
                    api_key=self.api_key,
                    use_cache=self.use_cache,
                )
                if dem is not None and np.isfinite(dem).any():
                    self.dem_type = dem_type
                    break
                errors.append(f"{dem_type}: DEM без валидных значений")
            except Exception as exc:
                errors.append(f"{dem_type}: {exc}")

        if dem is None or not np.isfinite(dem).any():
            raise ValueError(
                "Не удалось загрузить DEM для этой области. "
                "Пробованы источники: "
                + ", ".join(dem_candidates)
                + ". "
                + " | ".join(errors[:3])
            )

        self._bounds = (w, s, e, n)
        dem_smooth = median_filter(np.nan_to_num(dem, nan=0.0), size=MEDIAN_FILTER_SIZE)
        nan_mask = ~np.isfinite(dem)
        if np.any(nan_mask):
            dem_smooth[nan_mask] = np.nan
        self._elevation = dem_smooth
        return self._elevation

    def calculate_slope(self) -> np.ndarray:
        """Вычисляет уклон в градусах (метод Хорна). Требует уже загруженный DEM."""
        if self._elevation is None or self._bounds is None:
            raise RuntimeError("Сначала вызовите load_dem()")
        w, s, e, n = self._bounds
        rows, cols = self._elevation.shape
        # Размер ячейки в градусах → в метрах (для метода Хорна)
        lat_center_rad = np.radians((n + s) / 2.0)
        m_per_deg_lon = 111320 * np.cos(lat_center_rad)
        m_per_deg_lat = 110540
        cell_size_x_m = (e - w) / cols * m_per_deg_lon if cols else 1.0
        cell_size_y_m = (n - s) / rows * m_per_deg_lat if rows else 1.0
        self._slope = horn_slope_degrees(self._elevation, (cell_size_x_m, cell_size_y_m))
        return self._slope

    def create_visualization(
        self,
        lat: float,
        lon: float,
        title: Optional[str] = None,
    ) -> plt.Figure:
        """
        Строит карту рельефа: левая панель — высоты (terrain), правая — уклоны (coolwarm),
        с легендами и подписями координат. Возвращает Figure.
        """
        if self._elevation is None or self._bounds is None:
            self.load_dem(lat, lon)
        if self._slope is None:
            self.calculate_slope()

        w, s, e, n = self._bounds
        elev = self._elevation
        slope = self._slope

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

        # Карта высот
        im1 = ax1.imshow(
            elev,
            cmap="terrain",
            origin="upper",
            extent=[w, e, s, n],
            aspect="auto",
        )
        ax1.set_title("Карта высот", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Долгота (°)")
        ax1.set_ylabel("Широта (°)")
        plt.colorbar(im1, ax=ax1, shrink=0.7, label="Высота (м)")

        # Карта уклонов
        im2 = ax2.imshow(
            slope,
            cmap="coolwarm",
            origin="upper",
            extent=[w, e, s, n],
            aspect="auto",
        )
        ax2.set_title("Карта уклонов", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Долгота (°)")
        ax2.set_ylabel("Широта (°)")
        plt.colorbar(im2, ax=ax2, shrink=0.7, label="Уклон (°)")

        suptitle = title or f"Рельеф: {lat:.5f}°N, {lon:.5f}°E"
        fig.suptitle(suptitle, fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def build_map_png(
        self,
        lat: float,
        lon: float,
        output_path: os.PathLike,
        title: Optional[str] = None,
    ) -> str:
        """
        Полный цикл: загрузка DEM, фильтрация, уклон, визуализация, сохранение PNG.
        Возвращает путь к сохранённому файлу.
        """
        self.load_dem(lat, lon)
        self.calculate_slope()
        fig = self.create_visualization(lat, lon, title=title)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return str(path)

    def build_map_png_by_corners(
        self,
        left_top_lat: float,
        left_top_lon: float,
        right_bottom_lat: float,
        right_bottom_lon: float,
        output_path: os.PathLike,
        title: Optional[str] = None,
    ) -> str:
        self.load_dem_by_corners(
            left_top_lat=left_top_lat,
            left_top_lon=left_top_lon,
            right_bottom_lat=right_bottom_lat,
            right_bottom_lon=right_bottom_lon,
        )
        self.calculate_slope()
        fig = self.create_visualization(0.0, 0.0, title=title)
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return str(path)

    @staticmethod
    def _write_binary_stl(path: Path, triangles: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]]) -> None:
        """Сохраняет треугольники в бинарный STL."""
        header = b"ReliefAnalyzer STL".ljust(80, b"\0")
        with path.open("wb") as f:
            f.write(header)
            f.write(struct.pack("<I", len(triangles)))
            for v1, v2, v3 in triangles:
                # Нормаль можно не считать: большинство slicer/СAD пересчитает автоматически.
                f.write(struct.pack("<12fH", 0.0, 0.0, 0.0, *v1, *v2, *v3, 0))

    @staticmethod
    def _downsample_block_mean(arr: np.ndarray, block: int) -> np.ndarray:
        """Уменьшает DEM через усреднение по блокам, без aliasing от прореживания."""
        if block <= 1:
            return arr
        rows, cols = arr.shape
        new_rows = rows // block
        new_cols = cols // block
        if new_rows < 2 or new_cols < 2:
            return arr
        trimmed = arr[: new_rows * block, : new_cols * block]
        reshaped = trimmed.reshape(new_rows, block, new_cols, block)
        return np.nanmean(reshaped, axis=(1, 3))

    def build_relief_stl(
        self,
        lat: float,
        lon: float,
        output_path: os.PathLike,
        bbox_deg: Optional[float] = None,
        size_mm: float = 100.0,
        base_thickness_mm: float = 4.0,
        relief_height_mm: float = 16.0,
        smooth_sigma: float = 2.4,
        max_grid_size: int = 120,
        upscale_factor: float = 4.0,
        max_final_grid_size: int = 320,
        clip_percentile_low: float = 2.0,
        clip_percentile_high: float = 98.0,
    ) -> str:
        """
        Строит STL-модель рельефа фиксированного размера size_mm x size_mm
        с платформой-основанием (толщина base_thickness_mm).
        """
        dem = self.load_dem(lat, lon, bbox_deg=bbox_deg)
        dem = np.array(dem, dtype=float, copy=True)
        if dem.ndim != 2 or dem.shape[0] < 2 or dem.shape[1] < 2:
            raise ValueError("DEM слишком маленький для построения STL.")

        # Уменьшаем сетку через блоковое усреднение (а не прореживание по шагу),
        # чтобы убрать "частокол" и сохранить форму рельефа.
        rows, cols = dem.shape
        block = max(1, int(np.ceil(max(rows, cols) / max_grid_size)))
        dem = self._downsample_block_mean(dem, block)

        finite_mask = np.isfinite(dem)
        if not np.any(finite_mask):
            raise ValueError("DEM не содержит валидных высот.")

        # Заполняем пропуски локальным средним, чтобы не "ронять" точки вниз к минимуму.
        if not np.all(finite_mask):
            dem0 = np.where(finite_mask, dem, 0.0)
            weights = finite_mask.astype(float)
            local_sum = gaussian_filter(dem0, sigma=2.0, mode="nearest")
            local_w = gaussian_filter(weights, sigma=2.0, mode="nearest")
            local_mean = np.divide(local_sum, local_w, out=np.zeros_like(local_sum), where=local_w > 1e-9)
            dem = np.where(finite_mask, dem, local_mean)

        # Убираем экстремальные выбросы (часто дают "иглы" в урбанизированных зонах).
        p_low = float(np.percentile(dem, clip_percentile_low))
        p_high = float(np.percentile(dem, clip_percentile_high))
        if p_high > p_low:
            dem = np.clip(dem, p_low, p_high)

        # Мягкое сглаживание поверхности для более "печатного" рельефа без резких ступенек.
        if smooth_sigma > 0:
            dem = gaussian_filter(dem, sigma=smooth_sigma, mode="nearest")

        # Повышаем разрешение сетки интерполяцией (более гладкая геометрия STL).
        if upscale_factor > 1.0:
            current_rows, current_cols = dem.shape
            safe_factor = float(upscale_factor)
            max_dim = max(current_rows, current_cols)
            if max_dim * safe_factor > max_final_grid_size:
                safe_factor = max_final_grid_size / max_dim
            if safe_factor > 1.0:
                dem = zoom(dem, zoom=safe_factor, order=3, mode="nearest")
                dem = gaussian_filter(dem, sigma=1.2, mode="nearest")

        elev_range = float(np.max(dem) - np.min(dem))
        if elev_range < 1e-9:
            z_rel = np.zeros_like(dem, dtype=float)
        else:
            z_rel = (dem - np.min(dem)) / elev_range * relief_height_mm

        z_top = base_thickness_mm + z_rel

        rows, cols = z_top.shape
        x = np.linspace(0.0, size_mm, cols)
        y = np.linspace(0.0, size_mm, rows)

        triangles: list[tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]] = []

        def v(i: int, j: int, z_arr: np.ndarray) -> tuple[float, float, float]:
            return (float(x[j]), float(y[i]), float(z_arr[i, j]))

        # Верхняя рельефная поверхность.
        for i in range(rows - 1):
            for j in range(cols - 1):
                v00 = v(i, j, z_top)
                v01 = v(i, j + 1, z_top)
                v10 = v(i + 1, j, z_top)
                v11 = v(i + 1, j + 1, z_top)
                triangles.append((v00, v10, v11))
                triangles.append((v00, v11, v01))

        # Боковые стенки стенда.
        for j in range(cols - 1):
            # Передняя (y = 0)
            t0 = v(0, j, z_top)
            t1 = v(0, j + 1, z_top)
            b0 = (t0[0], t0[1], 0.0)
            b1 = (t1[0], t1[1], 0.0)
            triangles.append((t0, b0, b1))
            triangles.append((t0, b1, t1))

            # Задняя (y = size)
            t0 = v(rows - 1, j, z_top)
            t1 = v(rows - 1, j + 1, z_top)
            b0 = (t0[0], t0[1], 0.0)
            b1 = (t1[0], t1[1], 0.0)
            triangles.append((t0, b1, b0))
            triangles.append((t0, t1, b1))

        for i in range(rows - 1):
            # Левая (x = 0)
            t0 = v(i, 0, z_top)
            t1 = v(i + 1, 0, z_top)
            b0 = (t0[0], t0[1], 0.0)
            b1 = (t1[0], t1[1], 0.0)
            triangles.append((t0, b1, b0))
            triangles.append((t0, t1, b1))

            # Правая (x = size)
            t0 = v(i, cols - 1, z_top)
            t1 = v(i + 1, cols - 1, z_top)
            b0 = (t0[0], t0[1], 0.0)
            b1 = (t1[0], t1[1], 0.0)
            triangles.append((t0, b0, b1))
            triangles.append((t0, b1, t1))

        # Нижняя плоскость (дно стенда).
        p00 = (0.0, 0.0, 0.0)
        p10 = (size_mm, 0.0, 0.0)
        p11 = (size_mm, size_mm, 0.0)
        p01 = (0.0, size_mm, 0.0)
        triangles.append((p00, p11, p10))
        triangles.append((p00, p01, p11))

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_binary_stl(path, triangles)
        return str(path)

    def build_relief_stl_by_corners(
        self,
        left_top_lat: float,
        left_top_lon: float,
        right_bottom_lat: float,
        right_bottom_lon: float,
        output_path: os.PathLike,
        size_mm: float = 100.0,
        base_thickness_mm: float = 4.0,
        relief_height_mm: float = 16.0,
        smooth_sigma: float = 2.4,
        max_grid_size: int = 120,
        upscale_factor: float = 4.0,
        max_final_grid_size: int = 320,
        clip_percentile_low: float = 2.0,
        clip_percentile_high: float = 98.0,
    ) -> str:
        self.load_dem_by_corners(
            left_top_lat=left_top_lat,
            left_top_lon=left_top_lon,
            right_bottom_lat=right_bottom_lat,
            right_bottom_lon=right_bottom_lon,
        )
        if self._bounds is None:
            raise RuntimeError("Не удалось определить границы области.")
        w, s, e, n = self._bounds
        center_lat = (s + n) / 2.0
        center_lon = (w + e) / 2.0
        bbox_deg = max((n - s) / 2.0, (e - w) / 2.0)
        return self.build_relief_stl(
            lat=center_lat,
            lon=center_lon,
            output_path=output_path,
            bbox_deg=bbox_deg,
            size_mm=size_mm,
            base_thickness_mm=base_thickness_mm,
            relief_height_mm=relief_height_mm,
            smooth_sigma=smooth_sigma,
            max_grid_size=max_grid_size,
            upscale_factor=upscale_factor,
            max_final_grid_size=max_final_grid_size,
            clip_percentile_low=clip_percentile_low,
            clip_percentile_high=clip_percentile_high,
        )
