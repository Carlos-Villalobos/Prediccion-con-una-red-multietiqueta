"""Carga del dataset, definición de variables y preprocesado sin fugas."""
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
RUTA_DATOS = RAIZ / "MI.data"
SALIDAS = RAIZ / "salidas"

# Orden oficial de columnas (UCI, dataset 579). El fichero no trae cabecera.
COLUMNAS = (
    "ID AGE SEX INF_ANAM STENOK_AN FK_STENOK IBS_POST IBS_NASL GB SIM_GIPERT DLIT_AG ZSN_A "
    "nr_11 nr_01 nr_02 nr_03 nr_04 nr_07 nr_08 "
    "np_01 np_04 np_05 np_07 np_08 np_09 np_10 "
    "endocr_01 endocr_02 endocr_03 "
    "zab_leg_01 zab_leg_02 zab_leg_03 zab_leg_04 zab_leg_06 "
    "S_AD_KBRIG D_AD_KBRIG S_AD_ORIT D_AD_ORIT "
    "O_L_POST K_SH_POST MP_TP_POST SVT_POST GT_POST FIB_G_POST "
    "ant_im lat_im inf_im post_im IM_PG_P "
    "ritm_ecg_p_01 ritm_ecg_p_02 ritm_ecg_p_04 ritm_ecg_p_06 ritm_ecg_p_07 ritm_ecg_p_08 "
    "n_r_ecg_p_01 n_r_ecg_p_02 n_r_ecg_p_03 n_r_ecg_p_04 n_r_ecg_p_05 n_r_ecg_p_06 "
    "n_r_ecg_p_08 n_r_ecg_p_09 n_r_ecg_p_10 "
    "n_p_ecg_p_01 n_p_ecg_p_03 n_p_ecg_p_04 n_p_ecg_p_05 n_p_ecg_p_06 n_p_ecg_p_07 "
    "n_p_ecg_p_08 n_p_ecg_p_09 n_p_ecg_p_10 n_p_ecg_p_11 n_p_ecg_p_12 "
    "fibr_ter_01 fibr_ter_02 fibr_ter_03 fibr_ter_05 fibr_ter_06 fibr_ter_07 fibr_ter_08 "
    "GIPO_K K_BLOOD GIPER_NA NA_BLOOD ALT_BLOOD AST_BLOOD KFK_BLOOD L_BLOOD ROE TIME_B_S "
    "R_AB_1_n R_AB_2_n R_AB_3_n NA_KB NOT_NA_KB LID_KB NITR_S "
    "NA_R_1_n NA_R_2_n NA_R_3_n NOT_NA_1_n NOT_NA_2_n NOT_NA_3_n "
    "LID_S_n B_BLOK_S_n ANT_CA_S_n GEPAR_S_n ASP_S_n TIKL_S_n TRENT_S_n "
    "FIBR_PREDS PREDS_TAH JELUD_TAH FIBR_JELUD A_V_BLOK OTEK_LANC RAZRIV DRESSLER "
    "ZSN REC_IM P_IM_STEN LET_IS"
).split()

ENTRADAS = COLUMNAS[1:112]
COMPLICACIONES = COLUMNAS[112:123]


def _por_numero(*numeros):
    """Traduce números de columna de la documentación (base 1) a nombres."""
    return [COLUMNAS[n - 1] for n in numeros]


CONTINUAS = _por_numero(2, 35, 36, 37, 38, 84, 86, 87, 88, 89, 90, 91)
ORDINALES = _por_numero(
    4, 5, 6, 7, 9, 11, 45, 46, 47, 48, 92, 93, 94, 95, 100, 101, 102, 103, 104, 105
)
# ZSN_A es de orden parcial (0<1<2<4 y 0<1<3<4), así que no se trata como ordinal.
NOMINALES = _por_numero(12)
BINARIAS = [c for c in ENTRADAS if c not in CONTINUAS + ORDINALES + NOMINALES]

# Columnas no disponibles en cada momento, según la documentación de UCI.
_EXCLUIDAS = {
    "ingreso": _por_numero(93, 94, 95, 100, 101, 102, 103, 104, 105),
    "24h": _por_numero(94, 95, 101, 102, 104, 105),
    "48h": _por_numero(95, 102, 105),
    "72h": [],
}
# Variante propia: UCI admite los fármacos en UCI al ingreso, pero la documentación
# no fecha cuándo se administran. Sirve para medir cuánto dependen de ellos los resultados.
_EXCLUIDAS["ingreso_sin_uci"] = _EXCLUIDAS["ingreso"] + _por_numero(
    99, 106, 107, 108, 109, 110, 111, 112
)

# PREDS_TAH se descarta como salida: 20 positivos no permiten evaluarla.
SALIDAS_MODELO = ["MUERTE"] + [c for c in COMPLICACIONES if c != "PREDS_TAH"]

ETIQUETAS = {
    "MUERTE": "Muerte",
    "FIBR_PREDS": "Fibrilación auricular",
    "JELUD_TAH": "Taquicardia ventricular",
    "FIBR_JELUD": "Fibrilación ventricular",
    "A_V_BLOK": "Bloqueo AV de 3.er grado",
    "OTEK_LANC": "Edema pulmonar",
    "RAZRIV": "Rotura miocárdica",
    "DRESSLER": "Síndrome de Dressler",
    "ZSN": "Insuficiencia cardíaca crónica",
    "REC_IM": "Reinfarto",
    "P_IM_STEN": "Angina postinfarto",
}


def cargar_datos(ruta=RUTA_DATOS):
    datos = pd.read_csv(ruta, header=None, names=COLUMNAS, na_values="?")
    return datos


def columnas_momento(momento):
    return [c for c in ENTRADAS if c not in _EXCLUIDAS[momento]]


def objetivos(datos):
    """Matriz de salidas: muerte binarizada y las complicaciones evaluables."""
    Y = datos[[c for c in SALIDAS_MODELO if c != "MUERTE"]].copy()
    Y.insert(0, "MUERTE", (datos["LET_IS"] > 0).astype(int))
    return Y.astype(int)


def particiones_estratificadas(Y, k, semilla):
    """Asigna cada fila a una de k particiones con estratificación multietiqueta iterativa.

    Reparte primero la etiqueta más escasa (Sechidis et al., 2011), de modo que las
    complicaciones raras aparezcan en todas las particiones.
    """
    rng = np.random.default_rng(semilla)
    Y = np.asarray(Y, dtype=bool)
    n = len(Y)
    asignacion = np.full(n, -1)
    pendientes = np.ones(n, dtype=bool)
    deseo_total = np.full(k, n / k)
    deseo = np.tile(Y.sum(axis=0) / k, (k, 1))

    def asignar(fila, candidatas):
        candidatas = candidatas[deseo_total[candidatas] == deseo_total[candidatas].max()]
        destino = rng.choice(candidatas)
        asignacion[fila] = destino
        pendientes[fila] = False
        deseo[destino] -= Y[fila]
        deseo_total[destino] -= 1

    while Y[pendientes].any():
        restantes = Y[pendientes].sum(axis=0).astype(float)
        restantes[restantes == 0] = np.inf
        etiqueta = int(np.argmin(restantes))
        for fila in rng.permutation(np.flatnonzero(pendientes & Y[:, etiqueta])):
            maximo = deseo[:, etiqueta].max()
            asignar(fila, np.flatnonzero(deseo[:, etiqueta] == maximo))

    for fila in rng.permutation(np.flatnonzero(pendientes)):
        asignar(fila, np.arange(k))
    return asignacion


class Preprocesador:
    """Imputa, escala y codifica. Todo lo aprende en fit, solo con entrenamiento.

    Por defecto no añade indicadores de ausencia: en este dataset el hueco suele ser
    consecuencia del desenlace (sin analítica porque el paciente murió antes), lo que
    infla las métricas. umbral_indicador solo se activa para medir ese efecto.
    """

    def __init__(self, columnas, umbral_descarte=0.5, umbral_indicador=None):
        self.columnas = list(columnas)
        self.umbral_descarte = umbral_descarte
        self.umbral_indicador = umbral_indicador

    def fit(self, datos):
        ausentes = datos[self.columnas].isna().mean()
        utiles = [
            c for c in self.columnas
            if ausentes[c] <= self.umbral_descarte and datos[c].nunique() > 1
        ]
        self.descartadas_ = [c for c in self.columnas if c not in utiles]
        self.numericas_ = [c for c in utiles if c in CONTINUAS + ORDINALES]
        self.binarias_ = [c for c in utiles if c in BINARIAS]
        self.nominales_ = [c for c in utiles if c in NOMINALES]
        self.indicadores_ = []
        if self.umbral_indicador is not None:
            self.indicadores_ = [c for c in utiles if ausentes[c] >= self.umbral_indicador]

        self.medianas_ = datos[self.numericas_].median()
        imputadas = datos[self.numericas_].fillna(self.medianas_)
        self.medias_ = imputadas.mean()
        self.desviaciones_ = imputadas.std().replace(0, 1)
        self.modas_ = datos[self.binarias_ + self.nominales_].mode().iloc[0]
        self.categorias_ = {c: sorted(datos[c].dropna().unique()) for c in self.nominales_}

        self.nombres_ = (
            self.numericas_
            + self.binarias_
            + [f"{c}={v:g}" for c in self.nominales_ for v in self.categorias_[c]]
            + [f"{c}_ausente" for c in self.indicadores_]
        )
        return self

    def transform(self, datos):
        bloques = [
            (datos[self.numericas_].fillna(self.medianas_) - self.medias_) / self.desviaciones_,
            datos[self.binarias_].fillna(self.modas_[self.binarias_]),
        ]
        for columna in self.nominales_:
            valores = datos[columna].fillna(self.modas_[columna])
            bloques.append(pd.DataFrame(
                {v: (valores == v).astype(float) for v in self.categorias_[columna]},
                index=datos.index,
            ))
        bloques.append(datos[self.indicadores_].isna().astype(float))
        return np.hstack([b.to_numpy(dtype=np.float32) for b in bloques])

    def fit_transform(self, datos):
        return self.fit(datos).transform(datos)
