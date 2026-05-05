# Relief Analyzer

Инструмент для анализа рельефа по DEM:
- построение карты высот и уклонов (PNG);
- генерация 3D-модели рельефа (STL);
- Telegram-интерфейс для запуска обработки по координатам.

Источник DEM: [OpenTopography Global DEM API](https://portal.opentopography.org/apidocs/).

## Возможности

- загрузка DEM по области (2 угла) с fallback по источникам (`SRTMGL1`, `COP30`, `ASTER`);
- медианная фильтрация высот;
- расчёт уклона по Хорну (векторизованный `numpy`);
- отдельная быстрая реализация уклона в `fast_slope.py`;
- сравнение с эталоном `scipy` и бенчмарк производительности;
- экспорт PNG-карт и STL-моделей.

## Структура проекта

- `analyzer.py` — основной класс `ReliefAnalyzer`, загрузка DEM, уклон, PNG/STL.
- `fast_slope.py` — быстрая функция `calculate_slope_fast`.
- `compare_with_reference.py` — верификация уклона против `scipy.ndimage.convolve`.
- `benchmark_slope.py` — бенчмарк (`naive`, `legacy`, `fast`, `scipy`).
- `bot.py` — Telegram-бот (`/map`, `/stl`).
- `parse.py` — парсинг координат.
- `main.py` — точка входа.

## Установка

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Создай `.env`:

```bash
TELEGRAM_BOT_TOKEN=your_token
OPENTOPOGRAPHY_API_KEY=your_key
```

## Запуск

```bash
python -m main
```

или:

```bash
python main.py
```

## Использование как библиотеки

```python
from analyzer import ReliefAnalyzer

analyzer = ReliefAnalyzer(use_cache=True)
analyzer.build_map_png_by_corners(
    left_top_lat=55.80,
    left_top_lon=37.50,
    right_bottom_lat=55.70,
    right_bottom_lon=37.70,
    output_path="output/map.png",
)
```

## Верификация и бенчмарк

Сравнение с эталоном:

```bash
python compare_with_reference.py --dem-path path/to/dem.tif
```

Бенчмарк:

```bash
python benchmark_slope.py
```

Выходные файлы:
- `benchmark_results.csv`
- `plot_time_vs_size.png`
- `plot_speedup.png`
- `plot_error_map.png`

## Зависимости

- Python 3.11+
- `numpy`
- `scipy`
- `matplotlib`
- `rasterio`
- `requests`
- `aiogram`
- `python-dotenv`
