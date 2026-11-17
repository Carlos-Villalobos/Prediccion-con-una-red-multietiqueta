"""Comprueban la integridad del dataset y la ausencia de fugas en el preprocesado."""
import numpy as np
import pandas as pd
import pytest

from src.datos import (
    BINARIAS, COLUMNAS, COMPLICACIONES, CONTINUAS, ENTRADAS, NOMINALES, ORDINALES,
    Preprocesador, SALIDAS_MODELO, cargar_datos, columnas_momento, objetivos,
    particiones_estratificadas,
)
from src.entrenar import ajustar_pliegue


@pytest.fixture(scope="module")
def datos():
    return cargar_datos()


def test_dimensiones_e_identificador(datos):
    assert datos.shape == (1700, 124)
    assert len(COLUMNAS) == len(set(COLUMNAS)) == 124
    assert datos["ID"].is_unique


def test_estadisticos_coinciden_con_la_documentacion(datos):
    assert datos["SEX"].sum() == 1065
    assert datos["AGE"].isna().sum() == 8
    assert datos["IBS_NASL"].isna().sum() == 1628
    assert datos["KFK_BLOOD"].isna().sum() == 1696
    assert datos["K_BLOOD"].min() == 2.3 and datos["K_BLOOD"].max() == 8.2
    assert datos["NA_BLOOD"].min() == 117 and datos["NA_BLOOD"].max() == 169
    assert datos["ROE"].max() == 140


def test_tipos_cubren_todas_las_entradas(datos):
    assert sorted(BINARIAS + CONTINUAS + ORDINALES + NOMINALES) == sorted(ENTRADAS)
    assert len(ENTRADAS) == 111
    for columna in BINARIAS:
        assert set(datos[columna].dropna().unique()) <= {0, 1}, columna


def test_salidas_nunca_son_entradas():
    for momento in ("ingreso", "24h", "48h", "72h", "ingreso_sin_uci"):
        assert not set(columnas_momento(momento)) & set(COMPLICACIONES + ["LET_IS", "ID"])


def test_ingreso_excluye_informacion_posterior():
    ingreso = columnas_momento("ingreso")
    assert len(ingreso) == 102
    for columna in ("R_AB_1_n", "R_AB_2_n", "R_AB_3_n", "NA_R_2_n", "NOT_NA_3_n"):
        assert columna not in ingreso


def test_objetivos(datos):
    Y = objetivos(datos)
    assert list(Y.columns) == SALIDAS_MODELO
    assert Y["MUERTE"].sum() == 271
    assert "PREDS_TAH" not in Y


def test_particiones_cubren_todo_y_reparten_lo_raro(datos):
    Y = objetivos(datos)
    particion = particiones_estratificadas(Y, 5, semilla=0)
    assert set(particion) == {0, 1, 2, 3, 4}
    assert np.bincount(particion).max() - np.bincount(particion).min() <= 5
    for columna in Y:
        por_particion = Y[columna].groupby(particion).sum()
        assert por_particion.max() - por_particion.min() <= 2, columna


def test_particiones_son_reproducibles(datos):
    Y = objetivos(datos)
    a = particiones_estratificadas(Y, 5, semilla=3)
    b = particiones_estratificadas(Y, 5, semilla=3)
    assert (a == b).all()


def test_preprocesado_no_depende_de_validacion(datos):
    """Corromper las filas de validación no debe cambiar lo aprendido en el pliegue."""
    columnas = columnas_momento("ingreso")
    Y = objetivos(datos).to_numpy()
    entrada, validacion = np.arange(1000), np.arange(1000, 1700)
    config = {"ocultas": (8,), "dropout": 0.0, "lr": 1e-3, "weight_decay": 0.0,
              "pesado": False, "lote": 64}

    alterados = datos.copy()
    numericas = alterados.select_dtypes("number").columns.drop("ID")
    alterados.loc[alterados.index[validacion], numericas] *= 100

    limpio = ajustar_pliegue(datos, Y, entrada, validacion, columnas, config, 0, 1)["preprocesador"]
    corrupto = ajustar_pliegue(alterados, Y, entrada, validacion, columnas, config, 0, 1)["preprocesador"]
    assert limpio.nombres_ == corrupto.nombres_
    pd.testing.assert_series_equal(limpio.medias_, corrupto.medias_)
    pd.testing.assert_series_equal(limpio.medianas_, corrupto.medianas_)
    pd.testing.assert_series_equal(limpio.desviaciones_, corrupto.desviaciones_)


def test_por_defecto_no_usa_indicadores_de_ausencia(datos):
    """Los huecos delatan el desenlace; deben quedar fuera del modelo final."""
    preprocesador = Preprocesador(columnas_momento("ingreso")).fit(datos)
    assert preprocesador.indicadores_ == []
    assert not any(n.endswith("_ausente") for n in preprocesador.nombres_)


def test_preprocesado_descarta_vacias_y_no_deja_ausentes(datos):
    columnas = columnas_momento("ingreso")
    preprocesador = Preprocesador(columnas)
    X = preprocesador.fit_transform(datos)
    assert set(preprocesador.descartadas_) >= {"KFK_BLOOD", "IBS_NASL", "S_AD_KBRIG", "D_AD_KBRIG"}
    assert not np.isnan(X).any()
    assert X.shape == (1700, len(preprocesador.nombres_))
    assert "ZSN_A=4" in preprocesador.nombres_
