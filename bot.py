"""
Telegram-бот ReliefAnalyzer: команда «Построить карту высот и уклонов»,
пользователь отправляет координаты — бот возвращает PNG с картами высот и уклонов.
На базе aiogram (paper.md).
"""

import asyncio
import io
import logging
import os
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import BufferedInputFile, Message
from dotenv import load_dotenv

from parse import parse_bbox_coordinates
from analyzer import ReliefAnalyzer

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Директория для временных PNG (для отправки в Telegram)
OUTPUT_DIR = Path(__file__).resolve().parent / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
USER_MODES: dict[int, str] = {}


async def cmd_start(message: Message):
    """Приветствие и инструкция."""
    await message.answer(
        "🏔 <b>ReliefAnalyzer</b>\n\n"
        "Бот строит карты высот и уклонов по координатам с использованием DEM из OpenTopography.\n\n"
        "<b>Новое:</b>\n"
        "• <b>/stl</b> — 3D-модель рельефа в формате STL\n"
        "• Размер модели: <b>10x10 см</b>\n"
        "• Модель содержит <b>платформу-основание</b> (готовый стенд для печати)\n\n"
        "<b>Как использовать:</b>\n"
        "1. Для карты: команда <b>/map</b>\n"
        "2. Для 3D-модели: команда <b>/stl</b>\n"
        "3. Отправьте два угла области:\n"
        "   <code>левый_верхний_lat, левый_верхний_lon; правый_нижний_lat, правый_нижний_lon</code>\n"
        "   Например: <code>55.80, 37.50; 55.70, 37.70</code>\n"
        "4. Получите PNG-карту или STL-файл.\n\n"
        "Команды: /start, /map, /stl, /help",
        parse_mode="HTML",
    )


async def cmd_help(message: Message):
    """Справка по формату координат."""
    await message.answer(
        "<b>Формат координат области</b>\n"
        "• Широта: от -90 до 90 (север — положительные)\n"
        "• Долгота: от -180 до 180 (восток — положительные)\n"
        "• Передайте 2 точки: левый верхний и правый нижний углы\n"
        "• Разделители: запятая, пробел, точка с запятой\n"
        "• Пример: <code>55.80, 37.50; 55.70, 37.70</code>\n\n"
        "<b>Режимы:</b>\n"
        "• /map — PNG карта высот и уклонов\n"
        "• /stl — STL модель рельефа 10x10 см с платформой",
        parse_mode="HTML",
    )


async def cmd_map(message: Message):
    """Запрос координат для построения карты."""
    USER_MODES[message.chat.id] = "map"
    await message.answer(
        "📍 Введите два угла области:\n"
        "<code>левый_верхний_lat, левый_верхний_lon; правый_нижний_lat, правый_нижний_lon</code>\n\n"
        "Например: <code>55.80, 37.50; 55.70, 37.70</code>",
        parse_mode="HTML",
    )


async def cmd_stl(message: Message):
    """Запрос координат для построения STL-модели рельефа."""
    USER_MODES[message.chat.id] = "stl"
    await message.answer(
        "🧱 Введите два угла области:\n"
        "<code>левый_верхний_lat, левый_верхний_lon; правый_нижний_lat, правый_нижний_lon</code>\n\n"
        "Например: <code>55.80, 37.50; 55.70, 37.70</code>\n\n"
        "Я построю STL модели рельефа размером 10x10 см с основанием-платформой.",
        parse_mode="HTML",
    )


async def process_coordinates(message: Message):
    """Обработка текста с координатами: построение карты и отправка PNG."""
    if not message.text or not message.text.strip():
        return

    corners = parse_bbox_coordinates(message.text)
    if corners is None:
        await message.answer(
            "❌ Неверный формат координат.\n"
            "Нужно передать 2 угла области:\n"
            "левый верхний и правый нижний.\n"
            "Пример: 55.80, 37.50; 55.70, 37.70.\n"
            "Команда /help — подсказки.",
            parse_mode="HTML",
        )
        return

    lt_lat, lt_lon, rb_lat, rb_lon = corners
    north = max(lt_lat, rb_lat)
    south = min(lt_lat, rb_lat)
    west = min(lt_lon, rb_lon)
    east = max(lt_lon, rb_lon)
    mode = USER_MODES.get(message.chat.id, "map")
    if mode == "stl":
        status = await message.answer("🔄 Загружаю DEM и формирую STL-модель…")
    else:
        status = await message.answer("🔄 Загружаю DEM и строю карту…")

    def build_map_sync():
        analyzer = ReliefAnalyzer(use_cache=True)
        out_path = OUTPUT_DIR / f"relief_{north:.4f}_{west:.4f}_{south:.4f}_{east:.4f}.png"
        analyzer.build_map_png_by_corners(
            left_top_lat=lt_lat,
            left_top_lon=lt_lon,
            right_bottom_lat=rb_lat,
            right_bottom_lon=rb_lon,
            output_path=out_path,
            title=f"Рельеф области: LT({lt_lat:.4f}, {lt_lon:.4f}) RB({rb_lat:.4f}, {rb_lon:.4f})",
        )
        return out_path

    def build_stl_sync():
        analyzer = ReliefAnalyzer(use_cache=True)
        out_path = OUTPUT_DIR / f"relief_{north:.4f}_{west:.4f}_{south:.4f}_{east:.4f}.stl"
        analyzer.build_relief_stl_by_corners(
            left_top_lat=lt_lat,
            left_top_lon=lt_lon,
            right_bottom_lat=rb_lat,
            right_bottom_lon=rb_lon,
            output_path=out_path,
            size_mm=100.0,
            base_thickness_mm=4.0,
            relief_height_mm=16.0,
        )
        return out_path

    try:
        loop = asyncio.get_event_loop()
        if mode == "stl":
            out_path = await loop.run_in_executor(None, build_stl_sync)
        else:
            out_path = await loop.run_in_executor(None, build_map_sync)
    except Exception as e:
        logger.exception("Ошибка построения результата")
        await status.edit_text(
            f"❌ Ошибка при построении: {e}\n"
            "Проверьте координаты и повторите или используйте другой участок."
        )
        return

    if not out_path.exists():
        await status.edit_text("❌ Не удалось сохранить изображение.")
        return

    try:
        data = out_path.read_bytes()
        payload = BufferedInputFile(data, filename=out_path.name)
        if mode == "stl":
            await message.answer_document(
                payload,
                caption=(
                    "🧱 STL модель рельефа 10x10 см с платформой\n"
                    f"Область: LT({lt_lat:.5f}, {lt_lon:.5f}) RB({rb_lat:.5f}, {rb_lon:.5f})"
                ),
            )
            await status.edit_text("✅ STL готов.")
        else:
            await message.answer_photo(
                payload,
                caption=(
                    "🗺 Карта высот и уклонов\n"
                    f"Область: LT({lt_lat:.5f}, {lt_lon:.5f}) RB({rb_lat:.5f}, {rb_lon:.5f})"
                ),
            )
            await status.edit_text("✅ Готово.")
    except Exception as e:
        logger.exception("Ошибка отправки результата")
        await status.edit_text(f"❌ Ошибка отправки: {e}")
    finally:
        try:
            out_path.unlink()
        except OSError:
            pass


def setup_handlers(dp: Dispatcher):
    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_help, Command("help"))
    dp.message.register(cmd_map, Command("map"))
    dp.message.register(cmd_stl, Command("stl"))
    # Триггер по тексту команды из статьи
    dp.message.register(
        cmd_map,
        F.text.lower().in_({"построить карту высот и уклонов", "построить карту"}),
    )
    dp.message.register(
        cmd_stl,
        F.text.lower().in_({"построить stl", "сделать stl", "stl", "построить stl модель"}),
    )
    dp.message.register(process_coordinates, F.text)


async def main():
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        logger.error("TELEGRAM_BOT_TOKEN не задан. Создайте .env с TELEGRAM_BOT_TOKEN=...")
        return
    bot = Bot(token=token)
    dp = Dispatcher()
    setup_handlers(dp)
    logger.info("Бот запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
