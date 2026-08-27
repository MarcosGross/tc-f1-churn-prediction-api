# =========================================================================
# Dockerfile — API de Inferência de Churn (FastAPI + Scikit-Learn)
# Multi-stage build para imagem enxuta e segura (produção AWS)
# =========================================================================

# ---------- Stage 1: builder (compila dependências) ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Dependências de sistema apenas para compilar wheels (numpy/scipy/sklearn)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Constroi wheels pré-compiladas para acelerar a imagem final
RUN pip wheel --wheel-dir /wheels -r requirements.txt


# ---------- Stage 2: runtime (imagem final, mínima) ----------
FROM python:3.11-slim AS runtime

# Boas práticas: não rodar como root
RUN groupadd -r appuser && useradd -r -g appuser appuser

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app
ENV PYTHONPATH=/app/src

# Instala as wheels já compiladas (sem build-essential na imagem final)
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Copia código-fonte e o modelo treinado (.pkl / .joblib)
COPY src/ ./src/
COPY models/ ./models/

# Permissões para o usuário não-root
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Healthcheck nativo do Docker (usado também por ECS/App Runner)
##HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
    sys.exit(0) if urllib.request.urlopen('http://localhost:8000/health').status==200 else sys.exit(1)"

# Ajuste 'churn_prediction.api.main:app' conforme o caminho real do seu objeto FastAPI
CMD ["uvicorn", "churn_prediction.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
