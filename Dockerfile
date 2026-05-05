FROM python:3.11

WORKDIR /app

ENV PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY parse.py analyzer.py bot.py main.py ./

# Локальный кэш DEM (paper: Docker-контейнер с локальным кэшем)
VOLUME ["/app/cache", "/app/output"]

CMD ["python", "-m", "main"]
