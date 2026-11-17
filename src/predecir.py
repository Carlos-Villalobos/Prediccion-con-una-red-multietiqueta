"""Estima el riesgo de pacientes nuevos con el mismo flujo del modelo final.

Uso: python -m src.predecir ejemplos/pacientes_ejemplo.json

El fichero asocia un nombre a los datos de cada paciente. Lo que no se indique se
imputa como en el entrenamiento. Solo se aprende de desarrollo: el test no interviene.
"""
import argparse
import json

import numpy as np
import pandas as pd

from src.datos import (
    ETIQUETAS, SALIDAS, SALIDAS_MODELO, cargar_datos, columnas_momento, objetivos,
    particiones_estratificadas,
)
from src.entrenar import K, SEMILLA, calibrar, validacion_cruzada_mlp
from src.modelos import predecir_logits


def predecir(pacientes):
    """Devuelve el riesgo calibrado por paciente (filas) y desenlace (columnas)."""
    columnas = columnas_momento("ingreso")
    ajenas = sorted(set(pacientes.columns) - set(columnas))
    if ajenas:
        raise ValueError(f"Variables no disponibles al ingreso: {ajenas}")
    entradas = pacientes.reindex(columns=columnas).astype(float)

    resumen = json.loads((SALIDAS / "resumen_entrenamiento.json").read_text(encoding="utf-8"))
    config = {**resumen["config_multietiqueta"]}
    config["ocultas"] = tuple(config["ocultas"])

    datos = cargar_datos()
    Y = objetivos(datos).to_numpy()
    particion = particiones_estratificadas(Y, K, SEMILLA)
    test = np.flatnonzero(particion == 0)
    desarrollo = np.flatnonzero(particion != 0)

    logits_oof, _, resultados = validacion_cruzada_mlp(
        datos, Y, desarrollo, test, columnas, config, resumen["repeticiones_cv"], 300, -2)
    logits = np.mean([
        predecir_logits(r["modelo"], r["preprocesador"].transform(entradas)) for r in resultados
    ], axis=0)
    _, riesgo = calibrar(logits_oof, logits, Y, desarrollo)
    tabla = pd.DataFrame(riesgo, index=pacientes.index, columns=SALIDAS_MODELO)
    return tabla, pd.Series(Y[desarrollo].mean(axis=0), index=SALIDAS_MODELO)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("fichero", help="JSON con los datos de cada paciente")
    args = parser.parse_args()
    with open(args.fichero, encoding="utf-8") as f:
        pacientes = pd.DataFrame.from_dict(json.load(f), orient="index")

    riesgo, frecuencia = predecir(pacientes)
    tabla = riesgo.T.assign(frecuencia_desarrollo=frecuencia).rename(index=ETIQUETAS)
    print((tabla.astype(float) * 100).round(1).to_string())


if __name__ == "__main__":
    main()
