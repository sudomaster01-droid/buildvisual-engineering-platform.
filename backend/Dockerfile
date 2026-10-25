FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ARG INSTALL_LOCAL_AI=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system buildvisual \
    && useradd --system --gid buildvisual --home-dir /app buildvisual

WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY requirements-local-ai.txt /app/requirements-local-ai.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /app/requirements.txt \
    && if [ "$INSTALL_LOCAL_AI" = "true" ]; then \
        python -m pip install -r /app/requirements-local-ai.txt; \
    fi
COPY app /app/app
RUN mkdir -p /app/data /app/models && chown -R buildvisual:buildvisual /app

USER buildvisual
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/healthz', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
