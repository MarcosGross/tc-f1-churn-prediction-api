# =========================================================================
# Métricas Prometheus (Golden Signals) para a API FastAPI
# Baseado na Aula 07 (FIAP): latência, tráfego, erros -> SLIs/SLOs
# SLOs alvo sugeridos: p99 < 300ms  |  error rate < 0.5%
# =========================================================================
from fastapi import FastAPI, Request
from prometheus_client import Counter, Histogram, make_asgi_app

# Tráfego + Erros (Golden Signals)
REQS = Counter(
    "http_requests_total",
    "Total de requisições",
    ["method", "path", "code"],
)

# Latência — histograma com buckets calibrados à SLO (evite médias!)
LAT = Histogram(
    "http_request_latency_seconds",
    "Latência por rota",
    ["path"],
    buckets=[0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 5],  # 0.3s = alvo do p99
)


def setup_metrics(app: FastAPI):
    """Adiciona middleware de métricas e expõe /metrics para o Prometheus."""

    @app.middleware("http")
    async def _mw(request: Request, call_next):
        path = request.url.path
        with LAT.labels(path).time():
            resp = await call_next(request)
        REQS.labels(request.method, path, resp.status_code).inc()
        return resp

    # Endpoint /metrics — alvo de scrape do Prometheus / CloudWatch agent
    app.mount("/metrics", make_asgi_app())
    return app
