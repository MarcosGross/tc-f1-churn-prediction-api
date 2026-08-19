# =========================================================================
# Detecção de Data Drift com Evidently AI
# Baseado na Aula 07 (FIAP): compara distribuição de referência (treino)
# vs. produção (dados recentes). Gera relatório HTML + JSON de status.
# Rode como job agendado (ex.: EventBridge + ECS Task) em janelas horárias/diárias.
# =========================================================================
import pandas as pd
from evidently.report import Report
from evidently.metrics import DataDriftPreset

# ref  = amostra do dataset de treino (Telco Churn) — a "verdade" de referência
# cur  = features das requisições recentes capturadas em produção
ref = pd.read_parquet("data/ref.parquet")
cur = pd.read_parquet("data/current.parquet")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref, current_data=cur)

report.save_html("reports/drift.html")
with open("reports/drift.json", "w") as f:
    f.write(report.as_dict().__repr__())

# Métricas típicas de drift (PSI, KL, KS): dispare re-treino/alerta quando
# PSI > 0.2 ou o preset acusar drift em fração relevante das features.
print("Relatório de drift gerado em reports/drift.html")
