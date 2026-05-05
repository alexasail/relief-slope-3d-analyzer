# Быстрый старт — ReliefAnalyzer

Телеграм-бот для анализа DEM: карты высот и уклонов по координатам (OpenTopography, метод Хорна).

## 1. Установка

```bash
cd diplom
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Настройка

Создайте `.env` с токеном бота (получить у [@BotFather](https://t.me/BotFather)):

```bash
TELEGRAM_BOT_TOKEN=ваш_токен
# Опционально: ключ OpenTopography (по умолчанию — демо-ключ)
# OPENTOPOGRAPHY_API_KEY=ваш_ключ
```

## 3. Запуск бота

```bash
python -m main
```

или

```bash
python main.py
```

## 4. Использование

1. В Telegram: команда **Построить карту высот и уклонов** или `/map`
2. Отправьте координаты: `55.7558, 37.6176`
3. Получите PNG: слева — карта высот, справа — карта уклонов

Команды: `/start`, `/map`, `/help`.

## Docker

```bash
docker-compose up -d
```

## Структура

- `parse.py` — парсинг координат
- `analyzer.py` — ReliefAnalyzer (OpenTopography, медианная фильтрация, метод Хорна, PNG)
- `bot.py` — телеграм-бот (aiogram)
- `main.py` — точка входа

Подробнее — в README.md.
