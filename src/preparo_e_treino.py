"""
Pipeline de Triagem de Avaliações (Olist) com Abstenção Dupla e Calibração.
Autor: Davi C. Maciel
Descrição: Script de treinamento comparativo entre TF-IDF de Palavras (Modelo A) 
e Caracteres (Modelo B), utilizando validação temporal, isolamento estrito de 
clientes (zero leakage), calibração fora-da-dobra e limiares via Limite Inferior de Wilson.
"""

import math
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
import skops.io as sio
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

Z = 1.96

def wilson(k, n):
    """Calcula o intervalo de confiança de Wilson para penalizar precisão em amostras pequenas."""
    if n == 0: return 0.0, 1.0
    p, d = k / n, 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)

def ece(p, y, bins=10):
    """Calcula o Expected Calibration Error (ECE)."""
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))

def escolher_limiar(conf, acertou, alvo=0.95, min_n=50):
    """Encontra o menor limiar de confiança cujo limite inferior de Wilson atinge a meta (95%)."""
    for t in np.linspace(0.5, 0.999, 200):
        sel = conf >= t
        n, k = int(sel.sum()), int(acertou[sel].sum())
        if n >= min_n and wilson(k, n)[0] >= alvo:
            return float(t)
    return None

def decidir(p_neg, t_neg, t_pos):
    """Aplica abstenção dupla: 1 (Negativa Segura), 0 (Positiva Segura) ou -1 (Revisão Humana)."""
    d = np.full(len(p_neg), -1)
    if t_neg is not None: d[p_neg >= t_neg] = 1
    if t_pos is not None: d[(1 - p_neg) >= t_pos] = 0
    return d

def treinar_e_avaliar(tv: pd.DataFrame, teste: pd.DataFrame, vetorizador, alvo=0.95, seed=42):
    """Executa o pipeline completo de treino, calibração out-of-fold e avaliação cega."""
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=seed)
    i_tr, i_val = next(gss.split(tv, groups=tv["cliente"]))
    tr, val = tv.iloc[i_tr], tv.iloc[i_val]
    
    base = Pipeline([
        ("tfidf", vetorizador),
        ("clf", LogisticRegression(max_iter=1000, random_state=seed))
    ])
    
    # Calibração Out-of-Fold para evitar vazamento de dados de confiança
    folds = list(GroupKFold(n_splits=5).split(tr["texto"], tr["y"], groups=tr["cliente"]))
    cal = CalibratedClassifierCV(base, method="isotonic", cv=folds) 
    cal.fit(tr["texto"], tr["y"])
    
    pv, yv = cal.predict_proba(val["texto"])[:, 1], val["y"].to_numpy()
    t_neg = escolher_limiar(pv, yv == 1, alvo)
    t_pos = escolher_limiar(1 - pv, yv == 0, alvo)
    
    pt, yt = cal.predict_proba(teste["texto"])[:, 1], teste["y"].to_numpy()
    d = decidir(pt, t_neg, t_pos)
    
    auto = d != -1
    k, n = int((d[auto] == yt[auto]).sum()), int(auto.sum())
    lo, hi = wilson(k, n)
    auto_neg = d == 1
    k_neg, n_neg = int((d[auto_neg] == yt[auto_neg]).sum()), int(auto_neg.sum())
    
    return dict(
        t_neg=t_neg, t_pos=t_pos, n_total=len(yt), n_auto=n, 
        cob_global=n / len(yt) if len(yt) else 0, 
        prec_global=k / n if n else float("nan"), ic95=(lo, hi),
        cob_neg=n_neg / len(yt) if len(yt) else 0, 
        prec_neg=k_neg / n_neg if n_neg else float("nan"),
        ece_teste=ece(pt, yt), pt=pt, yt=yt, modelo_treinado=cal
    )

if __name__ == "__main__":
    print("Módulo de motor de triagem carregado. Para executar o pipeline completo, utilize o notebook do Kaggle associado.")