"""
Execute este script UMA VEZ para gerar o arquivo model.pkl
Comando: python train_and_save.py
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import pickle

# ── Ajuste o caminho do dataset ──────────────────────────────────────────────
DATASET_PATH = "dataset_queimada_processado.csv"

df = pd.read_csv(DATASET_PATH)

# Remover colunas desnecessárias
colunas_remover = ["IQAr_final", "FRP_Nivel_Alerta", "ano_original", "IQAr_Class"]
X = df.drop(columns=[c for c in colunas_remover if c in df.columns])
y = df["IQAr_final"]

# One-Hot Encoding no Bioma
X = pd.get_dummies(X, columns=["Bioma"], prefix="Bioma")

# Feature engineering
X["FRP_log"]          = np.log1p(X["FRP"])
X["deficit_hidrico"]  = X["temperatura_c"] / (X["umidade_relativa_percentual"] + 1)
X["mes_x_hora"]       = X["mes"] * X["hora"]

feature_names = X.columns.tolist()

# Alvo em log
y_log = np.log1p(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

# Treinar
print("Treinando Random Forest...")
model = RandomForestRegressor(n_estimators=60, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Avaliar
y_pred      = np.expm1(model.predict(X_test))
y_test_orig = np.expm1(y_test)
r2  = r2_score(y_test_orig, y_pred)
mae = mean_absolute_error(y_test_orig, y_pred)
print(f"R²  : {r2:.4f}")
print(f"MAE : {mae:.4f} pontos de IQAr")

# Salvar modelo + lista de features
with open("model.pkl", "wb") as f:
    pickle.dump({"model": model, "feature_names": feature_names}, f)

print("\n✅ model.pkl salvo com sucesso!")
print("Features:", feature_names)
