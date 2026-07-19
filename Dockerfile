FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TORCH_HOME=/opt/cavo-model-cache

WORKDIR /app

RUN apt-get update \
    && apt-get install --yes --no-install-recommends gosu ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src

RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu \
        "torch>=2.4,<3" "torchvision>=0.19,<1" \
    && pip install --no-cache-dir . \
    && mkdir -p "$TORCH_HOME" \
    && python -c "from torchvision.models import ResNet18_Weights,resnet18; resnet18(weights=ResNet18_Weights.DEFAULT)"

COPY docker-entrypoint.sh /usr/local/bin/cavo-entrypoint
COPY deployment /seed

RUN useradd --create-home --uid 10001 cavo \
    && mkdir -p /data/catalog /data/confirmed \
    && chown -R cavo:cavo /data /opt/cavo-model-cache \
    && chmod 755 /usr/local/bin/cavo-entrypoint

HEALTHCHECK --interval=60s --timeout=15s --start-period=45s --retries=3 \
    CMD cavo-validate-deployment --fast || exit 1

ENTRYPOINT ["cavo-entrypoint"]
CMD ["cavo-bot"]
