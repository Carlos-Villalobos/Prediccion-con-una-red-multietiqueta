"""Métricas finales sobre test, con intervalos bootstrap.

Uso: python -m src.evaluar

Los umbrales se fijan con las predicciones fuera de pliegue de desarrollo; test solo se
usa para medir.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from src.datos import ETIQUETAS, SALIDAS, SALIDAS_MODELO

N_BOOTSTRAP = 2000
SEMILLA = 2026


def umbral_youden(y, riesgo):
    fpr, tpr, umbrales = roc_curve(y, riesgo)
    return float(umbrales[np.argmax(tpr - fpr)])


def intervalo(valores):
    return np.nanpercentile(valores, [2.5, 97.5])


def bootstrap(y, riesgo, metrica, rng):
    n = len(y)
    valores = []
    for _ in range(N_BOOTSTRAP):
        muestra = rng.integers(n, size=n)
        if 0 < y[muestra].sum() < n:
            valores.append(metrica(y[muestra], riesgo[muestra]))
    return np.array(valores)


def metricas_salida(y, riesgo, y_dev, riesgo_dev, rng):
    umbral = umbral_youden(y_dev, riesgo_dev)
    prediccion = riesgo >= umbral
    auroc_ic = intervalo(bootstrap(y, riesgo, roc_auc_score, rng))
    auprc_ic = intervalo(bootstrap(y, riesgo, average_precision_score, rng))
    return {
        "positivos_test": int(y.sum()),
        "prevalencia": float(y.mean()),
        "auroc": roc_auc_score(y, riesgo), "auroc_inf": auroc_ic[0], "auroc_sup": auroc_ic[1],
        "auprc": average_precision_score(y, riesgo), "auprc_inf": auprc_ic[0], "auprc_sup": auprc_ic[1],
        "brier": brier_score_loss(y, riesgo),
        "umbral": umbral,
        "sensibilidad": float(prediccion[y == 1].mean()),
        "especificidad": float((~prediccion[y == 0]).mean()),
    }


def diferencia_auroc(y, a, b, rng):
    """Bootstrap pareado de AUROC(a) - AUROC(b) sobre los mismos pacientes."""
    n = len(y)
    diferencias = []
    for _ in range(N_BOOTSTRAP):
        m = rng.integers(n, size=n)
        if 0 < y[m].sum() < n:
            diferencias.append(roc_auc_score(y[m], a[m]) - roc_auc_score(y[m], b[m]))
    inferior, superior = intervalo(np.array(diferencias))
    return roc_auc_score(y, a) - roc_auc_score(y, b), inferior, superior


def curva_calibracion(y, riesgo, grupos=5):
    """Riesgo medio predicho frente a tasa observada, por quintiles de riesgo."""
    tramo = pd.qcut(pd.Series(riesgo), grupos, labels=False, duplicates="drop").to_numpy()
    return pd.DataFrame({
        "predicho": pd.Series(riesgo).groupby(tramo).mean(),
        "observado": pd.Series(y).groupby(tramo).mean(),
        "n": pd.Series(y).groupby(tramo).size(),
    })


def main():
    rng = np.random.default_rng(SEMILLA)
    predicciones = pd.read_csv(SALIDAS / "predicciones_modelos.csv")
    tablas, comparacion, calibracion = [], [], []

    for (modelo, salida), grupo in predicciones.groupby(["modelo", "salida"], sort=False):
        dev = grupo[grupo["particion"] == "desarrollo"]
        test = grupo[grupo["particion"] == "test"]
        fila = metricas_salida(
            test["y"].to_numpy(), test["riesgo"].to_numpy(),
            dev["y"].to_numpy(), dev["riesgo"].to_numpy(), rng,
        )
        fila.update({
            "modelo": modelo, "salida": salida,
            "auroc_oof": roc_auc_score(dev["y"], dev["riesgo"]),
            "auprc_oof": average_precision_score(dev["y"], dev["riesgo"]),
        })
        tablas.append(fila)

    metricas = pd.DataFrame(tablas)
    columnas = ["modelo", "salida"] + [c for c in metricas if c not in ("modelo", "salida")]
    metricas = metricas[columnas]
    metricas.to_csv(SALIDAS / "metricas_test.csv", index=False)

    ancho = predicciones[predicciones["particion"] == "test"].pivot_table(
        index=["ID", "salida", "y"], columns="modelo", values="riesgo").reset_index()
    for salida in SALIDAS_MODELO:
        grupo = ancho[ancho["salida"] == salida]
        y = grupo["y"].to_numpy()
        pares = [("mlp_multietiqueta", "logistica")]
        if salida == "MUERTE":
            pares += [("mlp_multietiqueta", "mlp_muerte"), ("mlp_muerte", "logistica")]
        for a, b in pares:
            diferencia, inferior, superior = diferencia_auroc(
                y, grupo[a].to_numpy(), grupo[b].to_numpy(), rng)
            comparacion.append({
                "salida": salida, "a": a, "b": b, "dif_auroc": diferencia,
                "inf": inferior, "sup": superior,
            })
        curva = curva_calibracion(y, grupo["mlp_multietiqueta"].to_numpy())
        curva.insert(0, "salida", salida)
        calibracion.append(curva)
    pd.DataFrame(comparacion).to_csv(SALIDAS / "comparacion_modelos.csv", index=False)
    pd.concat(calibracion).to_csv(SALIDAS / "calibracion_test.csv", index=False)

    vista = metricas.assign(salida=metricas["salida"].map(ETIQUETAS))
    print(vista[["modelo", "salida", "positivos_test", "auroc", "auroc_inf", "auroc_sup",
                 "auprc", "brier", "auroc_oof"]].round(3).to_string(index=False))
    print(pd.DataFrame(comparacion).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
