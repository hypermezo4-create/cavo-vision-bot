FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TORCH_HOME=/data/model-cache

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 cavo \
    && mkdir -p /data/catalog /data/confirmed /data/model-cache \
    && chown -R cavo:cavo /data

USER cavo
CMD ["cavo-bot"]

