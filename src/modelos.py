"""Redes neuronales del proyecto: MLP con cuerpo compartido y entrenamiento."""
import copy

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import nn


class MLP(nn.Module):
    """Perceptrón multicapa con una neurona de salida por cada objetivo."""

    def __init__(self, n_entradas, ocultas, n_salidas, dropout):
        super().__init__()
        capas = []
        anterior = n_entradas
        for neuronas in ocultas:
            capas += [nn.Linear(anterior, neuronas), nn.ReLU(), nn.Dropout(dropout)]
            anterior = neuronas
        self.cuerpo = nn.Sequential(*capas)
        self.salida = nn.Linear(anterior, n_salidas)

    def forward(self, x):
        return self.salida(self.cuerpo(x))

    def representacion(self, x):
        """Activaciones de la última capa oculta, usadas en la proyección visual."""
        return self.cuerpo(x)


def auroc_medio(Y, puntuaciones):
    """Media del AUROC por salida, ignorando salidas sin ambas clases."""
    valores = [
        roc_auc_score(Y[:, j], puntuaciones[:, j])
        for j in range(Y.shape[1]) if 0 < Y[:, j].sum() < len(Y)
    ]
    return float(np.mean(valores))


def predecir_logits(modelo, X):
    modelo.eval()
    with torch.no_grad():
        return modelo(torch.as_tensor(X, dtype=torch.float32)).numpy()


def entrenar_mlp(X_ent, Y_ent, X_val, Y_val, config, semilla, max_epocas=300, paciencia=25):
    """Entrena con parada temprana sobre el AUROC medio de validación.

    Sin X_val entrena exactamente max_epocas, lo que se usa para el modelo de la
    proyección visual, y devuelve max_epocas como mejor época.
    """
    torch.manual_seed(semilla)
    generador = torch.Generator().manual_seed(semilla)
    Y_ent = np.asarray(Y_ent, dtype=np.float32)
    modelo = MLP(X_ent.shape[1], config["ocultas"], Y_ent.shape[1], config["dropout"])
    optimizador = torch.optim.AdamW(
        modelo.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    peso = None
    if config["pesado"]:
        positivos = Y_ent.sum(axis=0)
        peso = torch.as_tensor(np.sqrt((len(Y_ent) - positivos) / positivos), dtype=torch.float32)
    criterio = nn.BCEWithLogitsLoss(pos_weight=peso)

    X = torch.as_tensor(X_ent, dtype=torch.float32)
    Y = torch.as_tensor(Y_ent)
    mejor_auroc, mejor_epoca, mejor_estado = -np.inf, max_epocas, None

    for epoca in range(1, max_epocas + 1):
        modelo.train()
        for lote in torch.randperm(len(X), generator=generador).split(config["lote"]):
            optimizador.zero_grad()
            criterio(modelo(X[lote]), Y[lote]).backward()
            optimizador.step()
        if X_val is None:
            continue
        auroc = auroc_medio(Y_val, predecir_logits(modelo, X_val))
        if auroc > mejor_auroc:
            mejor_auroc, mejor_epoca = auroc, epoca
            mejor_estado = copy.deepcopy(modelo.state_dict())
        elif epoca - mejor_epoca >= paciencia:
            break

    if mejor_estado is not None:
        modelo.load_state_dict(mejor_estado)
    return modelo, mejor_epoca
