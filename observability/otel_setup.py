# =========================================================================
# Instrumentação OpenTelemetry para a API FastAPI
# Baseado na Aula 07 (FIAP) — traces de latência p95/p99 do /predict
# Em AWS, exporte via OTLP para o ADOT Collector (CloudWatch / X-Ray).
# =========================================================================
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

# Para produção AWS, troque o ConsoleSpanExporter pelo OTLP:
# from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter


def setup_otel(app, service_name: str = "churn-api"):
    """Instrumenta a API FastAPI com OpenTelemetry (traces).

    Captura automaticamente spans HTTP (latência, status) de cada rota,
    incluindo /health e /predict — base para os SLIs de latência (p95/p99).
    """
    provider = TracerProvider(
        resource=Resource.create({"service.name": service_name})
    )

    # DEV/local: imprime spans no console.
    provider.add_span_processor(BatchSpanProcessor(ConsoleSpanExporter()))

    # PRODUÇÃO AWS: exporte via OTLP para o AWS Distro for OpenTelemetry (ADOT)
    # rodando como sidecar/coletor, que encaminha para CloudWatch/X-Ray:
    # otlp = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
    # provider.add_span_processor(BatchSpanProcessor(otlp))

    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    return app
