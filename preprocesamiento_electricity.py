"""
preprocesamiento_electricity.py
--------------------------------
Preprocesamiento del dataset UCI ElectricityLoadDiagrams20112014
para comparar RNN vs LSTM vs GRU (proyecto n>20, m>20.000).

CÓMO OBTENER LOS DATOS
-----------------------
1. Entra a: https://archive.ics.uci.edu/dataset/321/electricityloaddiagrams20112014
2. Descarga el .zip (botón "Download").
3. Descomprímelo. Dentro encontrarás un archivo llamado "LD2011_2014.txt"
   - separador: ";"
   - decimal: ","
   - primera columna: fecha/hora
   - resto de columnas: un cliente cada una, valores en kW cada 15 min

USO
---
    conda activate deep
    pip install pandas numpy scikit-learn matplotlib joblib
    python preprocesamiento_electricity.py --raw_path LD2011_2014.txt --out_dir data_procesada

SALIDA
------
    data_procesada/dataset_procesado.npz   -> X_train, y_train, X_val, y_val, X_test, y_test
    data_procesada/scaler.pkl              -> MinMaxScaler ajustado (para invertir predicciones)
    data_procesada/muestra_series.png      -> gráfico exploratorio
    Además imprime en consola toda la exploración: pégame esa salida.
"""

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

RANDOM_SEED = 42
N_CLIENTES = 25     # columnas/clientes a usar -> exige n>20, dejamos margen
WINDOW = 24         # horas pasadas usadas como entrada (1 día)
HORIZON = 1         # pasos a predecir (siguiente hora)
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
# TEST_FRAC = 0.15 (resto)


# ---------------------------------------------------------------------------
# 1. CARGA
# ---------------------------------------------------------------------------
def cargar_datos(raw_path):
    print(f"Cargando {raw_path} ...")
    df = pd.read_csv(raw_path, sep=";", decimal=",", index_col=0, parse_dates=True)
    print("Shape crudo (filas x columnas):", df.shape)
    return df


# ---------------------------------------------------------------------------
# 2. EXPLORACIÓN INICIAL (EDA)
# ---------------------------------------------------------------------------
def explorar(df):
    print("\n=== EXPLORACIÓN INICIAL ===")
    print("Rango de fechas:", df.index.min(), "->", df.index.max())
    print("N° columnas (clientes):", df.shape[1])
    print("N° filas (timestamps, cada 15 min):", df.shape[0])
    print("Valores nulos totales:", int(df.isnull().sum().sum()))

    resumen = df.describe().T[["mean", "std", "min", "max"]]
    print("\nEstadísticos generales (primeros 10 clientes):")
    print(resumen.head(10))

    pct_zeros = (df == 0).mean().sort_values()
    print("\nClientes más 'completos' (menor % de ceros):")
    print(pct_zeros.head(10))
    print("\nClientes más 'incompletos' (mayor % de ceros, probablemente inactivos/no creados aún):")
    print(pct_zeros.tail(10))

    return pct_zeros


# ---------------------------------------------------------------------------
# 3. LIMPIEZA + SELECCIÓN DE COLUMNAS
# ---------------------------------------------------------------------------
def limpiar_y_seleccionar(df, n_clientes=N_CLIENTES):
    # 3.1 Descartar 2011: muchos clientes aún no existían (todo el año en cero)
    df = df[df.index >= "2012-01-01"]
    print("\nShape tras eliminar 2011:", df.shape)

    # 3.2 Resamplear de 15 min a 1 hora (promedio) -> reduce tamaño sin perder señal
    df_h = df.resample("1h").mean()
    print("Shape tras resample horario:", df_h.shape)

    # 3.3 Elegir clientes "completos" (poco tiempo en cero) y con varianza real
    #     (evita clientes casi planos que no aportan nada al aprendizaje)
    pct_zeros = (df_h == 0).mean()
    varianza = df_h.var()
    candidatos = pct_zeros[pct_zeros < 0.01].index          # <1% de horas en cero
    candidatos = varianza[candidatos].sort_values(ascending=False).index
    seleccion = list(candidatos[:n_clientes])
    print(f"\nClientes seleccionados ({len(seleccion)}): {seleccion}")

    df_sel = df_h[seleccion].copy()

    # 3.4 Rellenar nulos remanentes (si los hay) por interpolación temporal
    n_nulos = int(df_sel.isnull().sum().sum())
    if n_nulos > 0:
        print(f"Rellenando {n_nulos} valores nulos con interpolación temporal...")
        df_sel = df_sel.interpolate(method="time").ffill().bfill()
    else:
        print("Sin valores nulos remanentes tras la selección.")

    return df_sel


# ---------------------------------------------------------------------------
# 4. GRÁFICO EXPLORATORIO
# ---------------------------------------------------------------------------
def graficar_ejemplos(df_sel, out_dir):
    plt.figure(figsize=(12, 5))
    for col in df_sel.columns[:5]:
        plt.plot(df_sel.index[: 24 * 14], df_sel[col].values[: 24 * 14], label=col, alpha=0.8)
    plt.legend()
    plt.title("Consumo horario - primeras 2 semanas (muestra de 5 clientes)")
    plt.xlabel("Fecha")
    plt.ylabel("kW")
    plt.tight_layout()
    path = os.path.join(out_dir, "muestra_series.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"Gráfico guardado en: {path}")


# ---------------------------------------------------------------------------
# 5. VENTANEO (sliding window) -> genera los ejemplos (X, y)
# ---------------------------------------------------------------------------
def crear_ventanas(df_sel, window=WINDOW, horizon=HORIZON):
    """
    Por cada cliente genera ejemplos independientes con ventana deslizante:
        X = últimas `window` horas de consumo
        y = siguiente valor de consumo
    Todos los clientes aportan (aprox.) el mismo número de ventanas -> dataset
    balanceado por cliente (nadie domina el entrenamiento).
    """
    X, y, client_id = [], [], []
    valores = df_sel.values  # (T, n_clientes)
    T, n_clientes = valores.shape

    for c in range(n_clientes):
        serie = valores[:, c]
        for t in range(window, T - horizon + 1):
            X.append(serie[t - window : t])
            y.append(serie[t : t + horizon])
            client_id.append(c)

    X = np.array(X)[..., np.newaxis]  # (m, window, 1) -> formato esperado por RNN/LSTM/GRU
    y = np.array(y)
    client_id = np.array(client_id)

    print(f"\nVentanas generadas: X={X.shape}, y={y.shape}")
    print(f"n (columnas/clientes usados)  = {n_clientes}  {'OK (>20)' if n_clientes > 20 else 'INSUFICIENTE'}")
    print(f"m (ejemplos totales generados) = {X.shape[0]}  {'OK (>20000)' if X.shape[0] > 20000 else 'INSUFICIENTE'}")

    # Chequeo de balance: ejemplos por cliente
    ejemplos_por_cliente = pd.Series(client_id).value_counts().sort_index()
    print("\nEjemplos por cliente (deben ser casi idénticos = balanceado):")
    print(ejemplos_por_cliente.describe())

    return X, y, client_id


# ---------------------------------------------------------------------------
# 6. SPLIT TEMPORAL (nunca mezclar tiempo dentro de un cliente)
# ---------------------------------------------------------------------------
def split_temporal(X, y, client_id, train_frac=TRAIN_FRAC, val_frac=VAL_FRAC):
    idx_train, idx_val, idx_test = [], [], []
    for c in np.unique(client_id):
        idx_c = np.where(client_id == c)[0]
        n = len(idx_c)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        idx_train.extend(idx_c[:n_train])
        idx_val.extend(idx_c[n_train : n_train + n_val])
        idx_test.extend(idx_c[n_train + n_val :])

    train = (X[idx_train], y[idx_train])
    val = (X[idx_val], y[idx_val])
    test = (X[idx_test], y[idx_test])
    return train, val, test


# ---------------------------------------------------------------------------
# 7. ESCALADO (fit SOLO con train, para evitar fuga de información)
# ---------------------------------------------------------------------------
def escalar(train, val, test):
    (Xtr, ytr), (Xva, yva), (Xte, yte) = train, val, test

    scaler = MinMaxScaler()
    scaler.fit(Xtr.reshape(-1, 1))

    def transform(X, y):
        Xs = scaler.transform(X.reshape(-1, 1)).reshape(X.shape)
        ys = scaler.transform(y.reshape(-1, 1)).reshape(y.shape)
        return Xs, ys

    Xtr, ytr = transform(Xtr, ytr)
    Xva, yva = transform(Xva, yva)
    Xte, yte = transform(Xte, yte)

    return (Xtr, ytr), (Xva, yva), (Xte, yte), scaler


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_path", required=True, help="Ruta a LD2011_2014.txt")
    parser.add_argument("--out_dir", default="data_procesada")
    parser.add_argument("--n_clientes", type=int, default=N_CLIENTES)
    parser.add_argument("--window", type=int, default=WINDOW)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    df = cargar_datos(args.raw_path)
    explorar(df)
    df_sel = limpiar_y_seleccionar(df, n_clientes=args.n_clientes)
    graficar_ejemplos(df_sel, args.out_dir)

    X, y, client_id = crear_ventanas(df_sel, window=args.window)
    train, val, test = split_temporal(X, y, client_id)
    train, val, test, scaler = escalar(train, val, test)

    np.savez(
        os.path.join(args.out_dir, "dataset_procesado.npz"),
        X_train=train[0], y_train=train[1],
        X_val=val[0], y_val=val[1],
        X_test=test[0], y_test=test[1],
    )

    import joblib
    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.pkl"))

    print("\n=== RESUMEN FINAL ===")
    print(f"n (clientes/columnas usados): {args.n_clientes}  (>20 requerido)")
    print(f"m (ejemplos totales):         {X.shape[0]}  (>20000 requerido)")
    print(f"Train: {train[0].shape[0]}  |  Val: {val[0].shape[0]}  |  Test: {test[0].shape[0]}")
    print(f"Archivos guardados en: {os.path.abspath(args.out_dir)}/")


if __name__ == "__main__":
    main()
