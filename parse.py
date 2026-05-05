"""
Парсинг координат из текстового сообщения пользователя.
Модуль parse — из архитектуры ReliefAnalyzer (paper.md).
"""

import re
from typing import Tuple, Optional


def parse_coordinates(text: str) -> Optional[Tuple[float, float]]:
    """
    Извлекает широту и долготу из строки.

    Поддерживаемые форматы:
    - 55.7558, 37.6176
    - 55.7558 37.6176
    - 55°45'21"N 37°37'02"E (градусы минуты секунды — упрощённо)
    - 55.7558,37.6176

    Args:
        text: Строка от пользователя.

    Returns:
        (lat, lon) или None при ошибке.
    """
    if not text or not text.strip():
        return None

    raw = text.strip().replace(",", " ").replace(";", " ")
    # Убираем лишние пробелы и разбиваем
    parts = re.split(r"\s+", raw)

    numbers: list[float] = []
    for p in parts:
        # Убираем возможные символы градусов и т.п.
        cleaned = re.sub(r"[°'\"NSEW]", "", p.replace(",", "."))
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue

    if len(numbers) >= 2:
        lat, lon = numbers[0], numbers[1]
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return (lat, lon)
    return None


def parse_bbox_coordinates(text: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Извлекает две точки области: левый верхний и правый нижний угол.

    Поддерживаемые форматы:
    - 55.80, 37.50; 55.70, 37.70
    - 55.80 37.50 55.70 37.70

    Returns:
        (left_top_lat, left_top_lon, right_bottom_lat, right_bottom_lon) или None.
    """
    if not text or not text.strip():
        return None

    raw = text.strip().replace(",", " ").replace(";", " ")
    parts = re.split(r"\s+", raw)

    numbers: list[float] = []
    for p in parts:
        cleaned = re.sub(r"[°'\"NSEW]", "", p.replace(",", "."))
        try:
            numbers.append(float(cleaned))
        except ValueError:
            continue

    if len(numbers) < 4:
        return None

    lt_lat, lt_lon, rb_lat, rb_lon = numbers[0], numbers[1], numbers[2], numbers[3]
    if not (
        -90 <= lt_lat <= 90
        and -180 <= lt_lon <= 180
        and -90 <= rb_lat <= 90
        and -180 <= rb_lon <= 180
    ):
        return None
    return (lt_lat, lt_lon, rb_lat, rb_lon)


def validate_coordinates(lat: float, lon: float) -> bool:
    """Проверка диапазонов широты и долготы."""
    return -90 <= lat <= 90 and -180 <= lon <= 180
