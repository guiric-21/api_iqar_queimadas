import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Carregar modelo ───────────────────────────────────────────────────────────
try:
    with open("model.pkl", "rb") as f:
        bundle = pickle.load(f)
    MODEL         = bundle["model"]
    FEATURE_NAMES = bundle["feature_names"]
except FileNotFoundError:
    raise RuntimeError("model.pkl não encontrado. Execute train_and_save.py primeiro.")

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="IQAr Queimadas — PE",
    description="API de previsão do Índice de Qualidade do Ar a partir de dados de queimadas em Pernambuco.",
    version="1.0.0",
)

# Libera CORS para o frontend do Lovable (e qualquer outra origem)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Schema de entrada ─────────────────────────────────────────────────────────
class QueimadasInput(BaseModel):
    bioma:            str   = Field(..., examples=["Caatinga"])
    frp:              float = Field(..., ge=0, description="Fire Radiative Power (MW)")
    latitude:         float = Field(..., examples=[-8.081])
    longitude:        float = Field(..., examples=[-35.471])
    precipitacao:     float = Field(..., ge=0, description="Precipitação (mm/dia)")
    temperatura:      float = Field(..., description="Temperatura (°C)")
    umidade:          int   = Field(..., ge=0, le=100, description="Umidade relativa (%)")
    direcao_vento:    int   = Field(..., ge=0, le=360, description="Direção do vento (grau)")
    velocidade_vento: float = Field(..., ge=0, description="Velocidade do vento (m/s)")
    ano:              int   = Field(..., examples=[2024])
    mes:              int   = Field(..., ge=1, le=12)
    dia:              int   = Field(..., ge=1, le=31)
    hora:             int   = Field(..., ge=0, le=23)

# ── Função de pré-processamento (espelha o notebook) ─────────────────────────
def preparar_features(data: QueimadasInput) -> pd.DataFrame:
    row = {
        "FRP":                          data.frp,
        "Latitude":                     data.latitude,
        "Longitude":                    data.longitude,
        "precipitacao_mmdia":           data.precipitacao,
        "temperatura_c":                data.temperatura,
        "umidade_relativa_percentual":  data.umidade,
        "vento_direcao_grau":           data.direcao_vento,
        "vento_velocidade_ms":          data.velocidade_vento,
        "ano":                          data.ano,
        "mes":                          data.mes,
        "dia":                          data.dia,
        "hora":                         data.hora,
        "Bioma_Caatinga":               1 if data.bioma == "Caatinga" else 0,
        "Bioma_Mata Atlântica":         1 if data.bioma == "Mata Atlântica" else 0,
    }

    # Feature engineering (idêntico ao notebook)
    row["FRP_log"]          = np.log1p(row["FRP"])
    row["deficit_hidrico"]  = row["temperatura_c"] / (row["umidade_relativa_percentual"] + 1)
    row["mes_x_hora"]       = row["mes"] * row["hora"]

    # Garantir a ordem exata das features do modelo
    df = pd.DataFrame([row])[FEATURE_NAMES]
    return df

# ── Endpoint de previsão ──────────────────────────────────────────────────────
@app.post("/predict")
def predict(data: QueimadasInput):
    try:
        X = preparar_features(data)
        iqar_log = MODEL.predict(X)[0]
        iqar     = float(np.expm1(iqar_log))
        iqar     = round(max(0.0, iqar), 2)  # nunca negativo

        # Classificação CONAMA 491/2018
        if iqar <= 40:
            categoria = "Boa"
        elif iqar <= 80:
            categoria = "Moderada"
        elif iqar <= 120:
            categoria = "Ruim"
        else:
            categoria = "Muito Ruim / Péssima"

        return {"iqar": iqar, "categoria": categoria}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/")
def health():
    return {"status": "ok", "modelo": "Random Forest Regressor", "r2": 0.9574}
