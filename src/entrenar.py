"""Búsqueda de hiperparámetros, validación cruzada y exportación de predicciones.

Uso: python -m src.entrenar [--rapido]

El conjunto de test solo se predice aquí; sus métricas se calculan en src/evaluar.py.
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from src.datos import (
    SALIDAS, SALIDAS_MODELO, Preprocesador, cargar_datos, columnas_momento, objetivos,
    particiones_estratificadas,
)
from src.modelos import auroc_medio, entrenar_mlp, predecir_logits

SEMILLA = 2026
K = 5
ESPACIO = {
    "ocultas": [(32,), (64,), (128,), (64, 32), (128, 64)],
    "dropout": [0.2, 0.4, 0.6],
    "lr": [3e-4, 1e-3, 3e-3],
    "weight_decay": [1e-4, 1e-3, 1e-2, 1e-1],
    "pesado": [False, True],
    "lote": [32, 64],
}
VALORES_C = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def sigmoide(x):
    return 1 / (1 + np.exp(-x))


def muestrear_configuraciones(n, semilla):
    rng = np.random.default_rng(semilla)
    configuraciones = []
    while len(configuraciones) < n:
        config = {k: v[rng.integers(len(v))] for k, v in ESPACIO.items()}
        if config not in configuraciones:
            configuraciones.append(config)
    return configuraciones


def ajustar_pliegue(datos, Y, entrada, validacion, columnas, config, semilla, max_epocas, test=None,
                    umbral_indicador=None):
    """Entrena una red en un pliegue. El preprocesado se ajusta solo con `entrada`."""
    torch.set_num_threads(1)
    preprocesador = Preprocesador(columnas, umbral_indicador=umbral_indicador).fit(datos.iloc[entrada])
    X_ent = preprocesador.transform(datos.iloc[entrada])
    X_val = preprocesador.transform(datos.iloc[validacion])
    modelo, epoca = entrenar_mlp(
        X_ent, Y[entrada], X_val, Y[validacion], config, semilla, max_epocas=max_epocas
    )
    logits_val = predecir_logits(modelo, X_val)
    logits_test = None
    if test is not None:
        logits_test = predecir_logits(modelo, preprocesador.transform(datos.iloc[test]))
    return {
        "validacion": validacion, "logits_val": logits_val, "logits_test": logits_test,
        "epoca": epoca, "modelo": modelo, "preprocesador": preprocesador,
    }


def buscar(datos, Y, desarrollo, columnas, configuraciones, pliegues, max_epocas, n_trabajos):
    """Puntúa cada configuración por AUROC medio de validación cruzada."""
    trabajos = [
        delayed(ajustar_pliegue)(
            datos, Y, desarrollo[pliegues != k], desarrollo[pliegues == k], columnas,
            config, SEMILLA + k, max_epocas,
        )
        for config in configuraciones for k in range(K)
    ]
    resultados = Parallel(n_jobs=n_trabajos)(trabajos)
    filas = []
    for i, config in enumerate(configuraciones):
        tramo = resultados[i * K:(i + 1) * K]
        aurocs = [auroc_medio(Y[r["validacion"]], r["logits_val"]) for r in tramo]
        filas.append({
            **config, "ocultas": str(config["ocultas"]),
            "auroc_medio": np.mean(aurocs), "auroc_dt": np.std(aurocs),
            "epoca_mediana": np.median([r["epoca"] for r in tramo]), "indice": i,
        })
    tabla = pd.DataFrame(filas).sort_values("auroc_medio", ascending=False)
    return tabla, configuraciones[int(tabla.iloc[0]["indice"])]


def validacion_cruzada_mlp(datos, Y, desarrollo, test, columnas, config, repeticiones,
                           max_epocas, n_trabajos, umbral_indicador=None):
    """CV repetida. Devuelve predicción fuera de pliegue y media de los modelos en test."""
    trabajos = []
    for repeticion in range(repeticiones):
        pliegues = particiones_estratificadas(Y[desarrollo], K, SEMILLA + 100 + repeticion)
        for k in range(K):
            trabajos.append(delayed(ajustar_pliegue)(
                datos, Y, desarrollo[pliegues != k], desarrollo[pliegues == k], columnas, config,
                SEMILLA + 1000 * (repeticion + 1) + k, max_epocas, test, umbral_indicador,
            ))
    resultados = Parallel(n_jobs=n_trabajos)(trabajos)
    logits_oof = np.zeros((len(datos), Y.shape[1]))
    for r in resultados:
        logits_oof[r["validacion"]] += r["logits_val"] / repeticiones
    logits_test = np.mean([r["logits_test"] for r in resultados], axis=0)
    return logits_oof, logits_test, resultados


def calibrar(logits_oof, logits_test, Y, desarrollo):
    """Escalado de Platt por salida, aprendido con las predicciones fuera de pliegue."""
    p_oof, p_test = np.zeros_like(logits_oof), np.zeros_like(logits_test)
    for j in range(Y.shape[1]):
        platt = LogisticRegression(C=1e6, max_iter=1000)
        platt.fit(logits_oof[desarrollo, j:j + 1], Y[desarrollo, j])
        p_oof[:, j] = platt.predict_proba(logits_oof[:, j:j + 1])[:, 1]
        p_test[:, j] = platt.predict_proba(logits_test[:, j:j + 1])[:, 1]
    return p_oof, p_test


def _logistica_pliegue(X_ent, y_ent, X_val, X_test, C):
    modelo = LogisticRegression(C=C, max_iter=3000).fit(X_ent, y_ent)
    return modelo.predict_proba(X_val)[:, 1], modelo.predict_proba(X_test)[:, 1]


def referencia_logistica(datos, Y, desarrollo, test, columnas, n_trabajos):
    """Una regresión logística por salida; C se elige por AUROC de validación cruzada."""
    pliegues = particiones_estratificadas(Y[desarrollo], K, SEMILLA)
    matrices = []
    for k in range(K):
        preprocesador = Preprocesador(columnas).fit(datos.iloc[desarrollo[pliegues != k]])
        matrices.append((
            k,
            preprocesador.transform(datos.iloc[desarrollo[pliegues != k]]),
            preprocesador.transform(datos.iloc[desarrollo[pliegues == k]]),
            preprocesador.transform(datos.iloc[test]),
        ))
    p_oof, p_test = np.zeros_like(Y, dtype=float), np.zeros((len(test), Y.shape[1]))
    elegidos = {}
    y_dev = Y[desarrollo]
    for j, salida in enumerate(SALIDAS_MODELO):
        trabajos = [
            delayed(_logistica_pliegue)(X_e, y_dev[pliegues != k, j], X_v, X_t, C)
            for C in VALORES_C for k, X_e, X_v, X_t in matrices
        ]
        resultados = Parallel(n_jobs=n_trabajos)(trabajos)
        puntuaciones = []
        for i in range(len(VALORES_C)):
            oof = np.zeros(len(desarrollo))
            for k in range(K):
                oof[pliegues == k] = resultados[i * K + k][0]
            puntuaciones.append((roc_auc_score(y_dev[:, j], oof), oof, i))
        _, oof, mejor = max(puntuaciones, key=lambda t: t[0])
        p_oof[desarrollo, j] = oof
        p_test[:, j] = np.mean([resultados[mejor * K + k][1] for k in range(K)], axis=0)
        elegidos[salida] = VALORES_C[mejor]
    return p_oof, p_test, elegidos


def _importancia_columna(columna, resultados, datos, Y, repeticiones, semilla):
    """Caída de AUROC fuera de pliegue por salida al permutar una columna original."""
    torch.set_num_threads(1)
    rng = np.random.default_rng(semilla)
    validacion = np.concatenate([r["validacion"] for r in resultados])
    base = np.concatenate([r["logits_val"] for r in resultados])
    caidas = np.zeros((repeticiones, Y.shape[1]))
    for repeticion in range(repeticiones):
        puntuaciones = []
        for r in resultados:
            permutado = datos.iloc[r["validacion"]].copy()
            permutado[columna] = rng.permutation(permutado[columna].to_numpy())
            puntuaciones.append(predecir_logits(r["modelo"], r["preprocesador"].transform(permutado)))
        puntuaciones = np.concatenate(puntuaciones)
        for j in range(Y.shape[1]):
            caidas[repeticion, j] = (
                roc_auc_score(Y[validacion, j], base[:, j])
                - roc_auc_score(Y[validacion, j], puntuaciones[:, j])
            )
    return caidas.mean(axis=0)


def importancia_permutacion(datos, Y, resultados, columnas, repeticiones, n_trabajos):
    """Usa solo la primera repetición de CV: nunca mira el conjunto de test."""
    presentes = [c for c in columnas if c not in resultados[0]["preprocesador"].descartadas_]
    filas = Parallel(n_jobs=n_trabajos)(
        delayed(_importancia_columna)(c, resultados, datos, Y, repeticiones, SEMILLA + i)
        for i, c in enumerate(presentes)
    )
    return pd.DataFrame(filas, index=presentes, columns=SALIDAS_MODELO).rename_axis("variable")


def experimento_momentos(datos, Y, desarrollo, config, repeticiones, max_epocas, n_trabajos):
    """Compara entradas en desarrollo para cuantificar las dos fuentes de fuga."""
    filas = []
    for nombre, momento, indicador in (
        ("ingreso_sin_uci", "ingreso_sin_uci", None),
        ("ingreso", "ingreso", None),
        ("ingreso_con_indicadores", "ingreso", 0.05),
        ("24h", "24h", None),
        ("48h", "48h", None),
        ("72h", "72h", None),
        ("72h_con_indicadores", "72h", 0.05),
    ):
        logits, _, _ = validacion_cruzada_mlp(
            datos, Y, desarrollo, desarrollo[:1], columnas_momento(momento), config,
            repeticiones, max_epocas, n_trabajos, indicador,
        )
        fila = {"momento": nombre}
        for j, salida in enumerate(SALIDAS_MODELO):
            fila[salida] = roc_auc_score(Y[desarrollo, j], logits[desarrollo, j])
        filas.append(fila)
    return pd.DataFrame(filas)


def proyeccion(datos, Y, desarrollo, columnas, config, epocas):
    """Proyección 2D de la capa oculta de un modelo entrenado con todo desarrollo."""
    torch.set_num_threads(1)
    preprocesador = Preprocesador(columnas).fit(datos.iloc[desarrollo])
    modelo, _ = entrenar_mlp(
        preprocesador.transform(datos.iloc[desarrollo]), Y[desarrollo], None, None,
        config, SEMILLA, max_epocas=epocas,
    )
    modelo.eval()
    with torch.no_grad():
        ocultas = modelo.representacion(torch.as_tensor(preprocesador.transform(datos))).numpy()
    pca = PCA(n_components=2, random_state=SEMILLA).fit(ocultas[desarrollo])
    return pca.transform(ocultas), pca.explained_variance_ratio_


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rapido", action="store_true", help="prueba corta del flujo completo")
    args = parser.parse_args()
    n_configs, repeticiones, max_epocas, rep_importancia = (
        (2, 1, 5, 1) if args.rapido else (24, 3, 300, 5)
    )
    n_trabajos = -2
    destino = SALIDAS / ("prueba" if args.rapido else "")
    destino.mkdir(parents=True, exist_ok=True)

    datos = cargar_datos()
    Y = objetivos(datos).to_numpy()
    particion = particiones_estratificadas(Y, K, SEMILLA)
    test = np.flatnonzero(particion == 0)
    desarrollo = np.flatnonzero(particion != 0)
    columnas = columnas_momento("ingreso")
    print(f"desarrollo={len(desarrollo)} test={len(test)} entradas={len(columnas)}", flush=True)

    pliegues = particiones_estratificadas(Y[desarrollo], K, SEMILLA + 1)
    configuraciones = muestrear_configuraciones(n_configs, SEMILLA)
    tabla_multi, config_multi = buscar(
        datos, Y, desarrollo, columnas, configuraciones, pliegues, max_epocas, n_trabajos)
    tabla_unica, config_unica = buscar(
        datos, Y[:, :1], desarrollo, columnas, configuraciones,
        particiones_estratificadas(Y[desarrollo, :1], K, SEMILLA + 1), max_epocas, n_trabajos)
    tabla_multi.to_csv(destino / "busqueda_multietiqueta.csv", index=False)
    tabla_unica.to_csv(destino / "busqueda_muerte.csv", index=False)
    print("mejor multietiqueta", config_multi, tabla_multi.iloc[0]["auroc_medio"], flush=True)
    print("mejor muerte", config_unica, tabla_unica.iloc[0]["auroc_medio"], flush=True)

    logits_oof, logits_test, resultados = validacion_cruzada_mlp(
        datos, Y, desarrollo, test, columnas, config_multi, repeticiones, max_epocas, n_trabajos)
    p_oof, p_test = calibrar(logits_oof, logits_test, Y, desarrollo)
    bruto_test = sigmoide(logits_test)

    logits_oof_u, logits_test_u, _ = validacion_cruzada_mlp(
        datos, Y[:, :1], desarrollo, test, columnas, config_unica, repeticiones, max_epocas,
        n_trabajos)
    pu_oof, pu_test = calibrar(logits_oof_u, logits_test_u, Y[:, :1], desarrollo)

    pl_oof, pl_test, valores_c = referencia_logistica(datos, Y, desarrollo, test, columnas, n_trabajos)
    print("logística C", valores_c, flush=True)

    importancia = importancia_permutacion(
        datos, Y, resultados[:K], columnas, rep_importancia, n_trabajos)
    importancia.to_csv(destino / "importancia_permutacion.csv")

    momentos = experimento_momentos(
        datos, Y, desarrollo, config_multi, repeticiones, max_epocas, n_trabajos)
    momentos.to_csv(destino / "momentos_prediccion.csv", index=False)
    print(momentos[["momento", "MUERTE"]].round(3).to_string(index=False), flush=True)

    epocas = max(1, int(np.median([r["epoca"] for r in resultados])))
    coordenadas, varianza = proyeccion(datos, Y, desarrollo, columnas, config_multi, epocas)

    # Predicciones por modelo. En desarrollo son fuera de pliegue; en test, media de modelos.
    filas = []
    for nombre, oof, en_test, salidas in (
        ("mlp_multietiqueta", p_oof, p_test, SALIDAS_MODELO),
        ("mlp_muerte", pu_oof, pu_test, ["MUERTE"]),
        ("logistica", pl_oof, pl_test, SALIDAS_MODELO),
    ):
        for j, salida in enumerate(salidas):
            for grupo, indices, valores in (
                ("desarrollo", desarrollo, oof[desarrollo, j]), ("test", test, en_test[:, j]),
            ):
                filas.append(pd.DataFrame({
                    "ID": datos["ID"].to_numpy()[indices], "particion": grupo, "modelo": nombre,
                    "salida": salida, "y": Y[indices, j], "riesgo": valores,
                }))
    pd.concat(filas).to_csv(destino / "predicciones_modelos.csv", index=False)

    # Tabla por paciente que alimenta la analítica visual.
    paciente = pd.DataFrame({"ID": datos["ID"], "particion": "desarrollo"})
    paciente.loc[test, "particion"] = "test"
    riesgos = p_oof.copy()
    riesgos[test] = p_test
    for j, salida in enumerate(SALIDAS_MODELO):
        paciente[f"riesgo_{salida}"] = riesgos[:, j]
        paciente[f"real_{salida}"] = Y[:, j]
    paciente["pc1"], paciente["pc2"] = coordenadas[:, 0], coordenadas[:, 1]
    paciente.to_csv(destino / "predicciones_oof.csv", index=False)

    resumen = {
        "semilla": SEMILLA, "n_desarrollo": len(desarrollo), "n_test": len(test),
        "config_multietiqueta": config_multi, "config_muerte": config_unica,
        "epoca_mediana": epocas, "repeticiones_cv": repeticiones,
        "entradas_finales": len(resultados[0]["preprocesador"].nombres_),
        "descartadas": resultados[0]["preprocesador"].descartadas_,
        "indicadores_ausencia": resultados[0]["preprocesador"].indicadores_,
        "varianza_pca": varianza.tolist(), "C_logistica": valores_c,
        "brier_test_sin_calibrar_media": float(np.mean((bruto_test - Y[test]) ** 2)),
    }
    (destino / "resumen_entrenamiento.json").write_text(
        json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    print("listo", flush=True)


if __name__ == "__main__":
    main()
