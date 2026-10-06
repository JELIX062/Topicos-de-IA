"""
Clasificación de galaxias con UN AUTOENCODER POR CLASE.

Idea:
  * Se entrenan tres autoencoders, cada uno SOLO con imágenes de su clase:
        AE-Elliptical, AE-Espiral y AE-Lenticular.
    Cada uno aprende a reconstruir bien "su" tipo de galaxia y mal las demás.
  * Cada autoencoder funciona como un CLASIFICADOR BINARIO
    ("¿esta imagen pertenece a mi clase, sí o no?") con un umbral sobre el
    error de reconstrucción.
  * Para clasificar una imagen nueva se pasa por los tres autoencoders, se
    compara la SALIDA (reconstrucción) con la ENTRADA (error MSE) y se asigna
    la clase del autoencoder con el error MÁS PEQUEÑO (el más cercano).

Uso:
    python clasificador_autoencoders_por_clase.py
    python clasificador_autoencoders_por_clase.py --epocas 30 --criterio normalizado

Las figuras se guardan en la carpeta  resultados_por_clase/
"""

import argparse
import glob
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf
from keras import layers

SEMILLA = 42
PORC_PRUEBA = 0.20  # 80% entrenamiento / 20% prueba
CARPETA_SALIDA = "resultados_por_clase"
PASOS_POR_EPOCA = 30  # mismo número de actualizaciones para los 3 autoencoders
COLORES = ["#1f77b4", "#ff7f0e", "#2ca02c"]


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def cargar_imagen(ruta, tam):
    """Lee una imagen en escala de grises, recorta el centro y la normaliza a [0, 1]."""
    img = Image.open(ruta).convert("L")
    img = ImageOps.fit(img, (tam, tam), Image.LANCZOS)
    return np.asarray(img, dtype="float32")[..., None] / 255.0


def cargar_datos(carpeta, tam):
    clases = sorted(d for d in os.listdir(carpeta) if os.path.isdir(os.path.join(carpeta, d)))
    X, y, rutas = [], [], []
    for i, clase in enumerate(clases):
        for ruta in sorted(glob.glob(os.path.join(carpeta, clase, "*"))):
            X.append(cargar_imagen(ruta, tam))
            y.append(i)
            rutas.append(ruta)
    return np.stack(X), np.array(y), np.array(rutas), clases


def aumentar(X):
    """Las galaxias no tienen 'arriba' ni 'abajo': se generan las 8 rotaciones/espejos."""
    Xs = []
    for k in range(4):
        rot = np.rot90(X, k, axes=(1, 2))
        Xs += [rot, rot[:, :, ::-1]]
    return np.concatenate(Xs)


# ---------------------------------------------------------------------------
# Autoencoders
# ---------------------------------------------------------------------------
def construir_autoencoder(forma, latente, nombre):
    """Autoencoder convolucional con un cuello de botella pequeño.

    El cuello de botella (vector de `latente` números) obliga a la red a quedarse
    solo con lo típico de SU clase; por eso reconstruye peor las otras clases.
    """
    entrada = keras.Input(forma, name="entrada")
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(entrada)
    x = layers.MaxPooling2D()(x)                                           # 32x32
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)                                           # 16x16
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)                                           # 8x8
    x = layers.Flatten()(x)
    codigo = layers.Dense(latente, activation="relu", name="codigo_latente")(x)

    x = layers.Dense(8 * 8 * 32, activation="relu")(codigo)
    x = layers.Reshape((8, 8, 32))(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)   # 16x16
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)   # 32x32
    x = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)   # 64x64
    salida = layers.Conv2D(forma[-1], 3, padding="same", activation="sigmoid", name="reconstruccion")(x)

    ae = keras.Model(entrada, salida, name=nombre)
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae


def error_reconstruccion(ae, X):
    """MSE por imagen entre la entrada y la salida del autoencoder."""
    rec = ae.predict(X, verbose=0, batch_size=64)
    return np.mean((X - rec) ** 2, axis=(1, 2, 3)), rec


def clasificar(autoencoders, X, escala=None):
    """Pasa cada imagen por TODOS los autoencoders y elige el de menor error.

    escala=None  -> se compara el MSE tal cual.
    escala=array -> se divide el MSE de cada autoencoder entre su error típico, para que un
                    autoencoder que reconstruye "todo bien" no gane siempre.
    """
    errores = np.stack([error_reconstruccion(ae, X)[0] for ae in autoencoders], axis=1)  # (n, n_clases)
    puntaje = errores if escala is None else errores / escala
    return puntaje.argmin(axis=1), errores, puntaje


# ---------------------------------------------------------------------------
# Gráficas
# ---------------------------------------------------------------------------
def guardar(fig, nombre):
    ruta = os.path.join(CARPETA_SALIDA, nombre)
    fig.savefig(ruta, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {ruta}")


def graficar_entrenamiento(historias, clases, nombre):
    fig, ejes = plt.subplots(1, len(clases), figsize=(5 * len(clases), 3.8), sharey=True)
    for eje, h, clase, color in zip(ejes, historias, clases, COLORES):
        eje.plot(h["loss"], color=color, label="entrenamiento")
        eje.plot(h["val_loss"], "--", color=color, alpha=0.7, label=f"prueba ({clase})")
        eje.set_title(f"AE-{clase}")
        eje.set_xlabel("época")
        eje.set_yscale("log")
        eje.grid(alpha=0.3)
        eje.legend(fontsize=8)
    ejes[0].set_ylabel("error de reconstrucción (MSE)")
    fig.suptitle("Entrenamiento: cada autoencoder aprende SOLO con imágenes de su clase", fontsize=13)
    fig.tight_layout()
    guardar(fig, nombre)


def graficar_entrada_vs_salida(X, y, autoencoders, clases, nombre, escala=None):
    """Para cada imagen: entrada, la reconstrucción de cada AE (salida) y su error."""
    n = len(X)
    k = len(clases)
    fig, ejes = plt.subplots(n, k + 2, figsize=(2.6 * (k + 2), 2.7 * n), squeeze=False,
                             gridspec_kw={"width_ratios": [1] * (k + 1) + [1.3]})
    recs = [ae.predict(X, verbose=0) for ae in autoencoders]
    for i in range(n):
        errores = [np.mean((X[i] - r[i]) ** 2) for r in recs]
        gana = int(np.argmin(errores if escala is None else np.array(errores) / escala))
        ejes[i, 0].imshow(X[i, ..., 0], cmap="gray", vmin=0, vmax=1)
        ejes[i, 0].set_title(f"ENTRADA\nreal: {clases[y[i]]}", fontsize=9)
        for j in range(k):
            eje = ejes[i, j + 1]
            eje.imshow(recs[j][i, ..., 0], cmap="gray", vmin=0, vmax=1)
            eje.set_title(f"SALIDA AE-{clases[j]}\nMSE = {errores[j]:.4f}", fontsize=9,
                          weight="bold" if j == gana else "normal")
            if j == gana:
                eje.add_patch(Rectangle((0, 0), 1, 1, transform=eje.transAxes, fill=False,
                                        ec="green" if gana == y[i] else "red", lw=5))
        for eje in ejes[i, :k + 1]:
            eje.set_xticks([])
            eje.set_yticks([])
        barras = ejes[i, -1]
        barras.barh(range(k), errores, color=COLORES[:k])
        barras.set_yticks(range(k), [f"AE-{c}" for c in clases], fontsize=8)
        barras.invert_yaxis()
        barras.tick_params(axis="x", labelsize=7)
        barras.set_title(f"Predicción: {clases[gana]}", fontsize=10, weight="bold",
                         color="green" if gana == y[i] else "red")
    fig.suptitle("Entrada vs. salida de cada autoencoder: gana el que reconstruye con MENOR error\n"
                 "(marco verde = acierto, rojo = error)", fontsize=13)
    fig.tight_layout()
    guardar(fig, nombre)


def graficar_mapas_error(X, y, autoencoders, clases, nombre):
    """|entrada - salida| de cada autoencoder: dónde 'no entiende' la imagen."""
    n, k = len(X), len(clases)
    fig, ejes = plt.subplots(n, k + 1, figsize=(2.4 * (k + 1), 2.5 * n), squeeze=False)
    recs = [ae.predict(X, verbose=0) for ae in autoencoders]
    vmax = max(np.abs(X - r).max() for r in recs) * 0.6
    for i in range(n):
        ejes[i, 0].imshow(X[i, ..., 0], cmap="gray")
        ejes[i, 0].set_title(f"real: {clases[y[i]]}", fontsize=9)
        for j in range(k):
            ejes[i, j + 1].imshow(np.abs(X[i, ..., 0] - recs[j][i, ..., 0]), cmap="inferno", vmin=0, vmax=vmax)
            ejes[i, j + 1].set_title(f"|entrada − AE-{clases[j]}|", fontsize=9)
        for eje in ejes[i]:
            eje.axis("off")
    fig.suptitle("Mapas de error: zonas brillantes = lo que el autoencoder NO pudo reconstruir", fontsize=13)
    fig.tight_layout()
    guardar(fig, nombre)


def graficar_matriz_errores(errores, y, clases, nombre, titulo_color="MSE (más claro = más cercano)"):
    """Error de cada imagen de prueba en cada autoencoder (filas ordenadas por clase real)."""
    orden = np.argsort(y, kind="stable")
    E = errores[orden]
    fig, eje = plt.subplots(figsize=(6, 0.28 * len(E) + 1.5))
    im = eje.imshow(E, cmap="viridis_r", aspect="auto")
    for i, fila in enumerate(E):
        j = fila.argmin()
        eje.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                ec="lime" if j == y[orden][i] else "red", lw=2))
    limites = np.cumsum([np.sum(y == c) for c in range(len(clases))])[:-1]
    for b in limites:
        eje.axhline(b - 0.5, color="white", lw=2)
    centros = [np.mean(np.where(y[orden] == c)[0]) for c in range(len(clases))]
    eje.set_yticks(centros, [f"real: {c}" for c in clases])
    eje.set_xticks(range(len(clases)), [f"AE-{c}" for c in clases])
    eje.set_title("Error de reconstrucción de cada imagen de prueba\n"
                  "cuadro = autoencoder elegido (verde = correcto, rojo = incorrecto)", fontsize=11)
    fig.colorbar(im, ax=eje, label=titulo_color)
    fig.tight_layout()
    guardar(fig, nombre)


def graficar_matriz_confusion(y_real, y_pred, clases, nombre):
    cm = confusion_matrix(y_real, y_pred, labels=range(len(clases)))
    fig, eje = plt.subplots(figsize=(5, 4.5))
    im = eje.imshow(cm, cmap="Blues")
    for i in range(len(clases)):
        for j in range(len(clases)):
            eje.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    eje.set_xticks(range(len(clases)), clases, rotation=20)
    eje.set_yticks(range(len(clases)), clases)
    eje.set_xlabel("Predicción (autoencoder con menor error)")
    eje.set_ylabel("Real")
    eje.set_title(f"Clasificación final (3 autoencoders)\nexactitud = {cm.trace() / cm.sum():.1%}")
    fig.colorbar(im, ax=eje, fraction=0.046)
    guardar(fig, nombre)


def graficar_binarios(err_prueba, y_prueba, umbrales, clases, nombre):
    """Cada autoencoder como clasificador binario: '¿pertenece a mi clase?'."""
    k = len(clases)
    fig, ejes = plt.subplots(2, k, figsize=(5.2 * k, 8), gridspec_kw={"height_ratios": [1.3, 1]})
    resumen = []
    for c in range(k):
        e = err_prueba[:, c]
        propia, otras = e[y_prueba == c], e[y_prueba != c]
        eje = ejes[0, c]
        bins = np.linspace(e.min(), e.max(), 20)
        eje.hist(otras, bins=bins, alpha=0.6, color="gray", label="otras clases")
        eje.hist(propia, bins=bins, alpha=0.8, color=COLORES[c], label=f"{clases[c]}")
        eje.axvline(umbrales[c], color="red", ls="--", label=f"umbral = {umbrales[c]:.4f}")
        auc = roc_auc_score(y_prueba == c, -e)
        eje.set_title(f"AE-{clases[c]}  (AUC = {auc:.2f})")
        eje.set_xlabel("error de reconstrucción (MSE)")
        eje.legend(fontsize=8)

        real = (y_prueba == c).astype(int)
        pred = (e <= umbrales[c]).astype(int)
        cm = confusion_matrix(real, pred, labels=[1, 0])
        eje = ejes[1, c]
        eje.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                eje.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        etiquetas = [f"Sí es {clases[c]}", "No es"]
        eje.set_xticks([0, 1], etiquetas)
        eje.set_yticks([0, 1], etiquetas)
        eje.set_xlabel("Predicción (error ≤ umbral)")
        eje.set_ylabel("Real")
        acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
        eje.set_title(f"Binario {clases[c]} vs. resto: exactitud = {acc:.1%}")
        resumen.append((clases[c], umbrales[c], auc, acc, cm))
    fig.suptitle("Cada autoencoder como CLASIFICADOR BINARIO: si el error es menor que el umbral, "
                 "'pertenece a mi clase'", fontsize=13)
    fig.tight_layout()
    guardar(fig, nombre)
    return resumen


def graficar_predicciones_prueba(X, y, pred, errores, clases, nombre, columnas=10):
    n = len(X)
    filas = int(np.ceil(n / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(1.9 * columnas, 2.5 * filas), squeeze=False)
    for k, eje in enumerate(ejes.flat):
        eje.axis("off")
        if k >= n:
            continue
        eje.imshow(X[k, ..., 0], cmap="gray")
        eje.set_title(f"Real: {clases[y[k]]}", fontsize=8)
        eje.text(0.5, -0.05, f"Pred: {clases[pred[k]]}", transform=eje.transAxes, ha="center", va="top",
                 fontsize=8, color="green" if pred[k] == y[k] else "red", weight="bold")
    fig.suptitle(f"20% de prueba ({n} imágenes): clase del autoencoder más cercano — "
                 f"exactitud {np.mean(pred == y):.1%}", fontsize=13)
    fig.subplots_adjust(hspace=0.5, wspace=0.08, top=1 - 0.6 / filas, bottom=0.2 / filas)
    guardar(fig, nombre)


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datos", default="data")
    p.add_argument("--tam", type=int, default=64)
    p.add_argument("--latente", type=int, default=32, help="tamaño del cuello de botella")
    p.add_argument("--epocas", type=int, default=20)
    p.add_argument("--criterio", choices=["mse", "normalizado"], default="mse",
                   help="mse: menor error tal cual | normalizado: error / error típico de cada autoencoder")
    args = p.parse_args()

    keras.utils.set_random_seed(SEMILLA)
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    print("Cargando imágenes...")
    X, y, rutas, clases = cargar_datos(args.datos, args.tam)
    X_ent, X_pru, y_ent, y_pru, r_ent, r_pru = train_test_split(
        X, y, rutas, test_size=PORC_PRUEBA, stratify=y, random_state=SEMILLA)
    print(f"  Entrenamiento (80%): {len(X_ent)}  |  Prueba (20%): {len(X_pru)}")

    # ===== 1) Un autoencoder por clase, entrenado SOLO con su clase =====
    autoencoders, historias, umbrales = [], [], []
    for c, clase in enumerate(clases):
        X_c = aumentar(X_ent[y_ent == c])
        print(f"\nEntrenando AE-{clase} con {np.sum(y_ent == c)} imágenes ({len(X_c)} con rotaciones/espejos)...")
        ae = construir_autoencoder(X.shape[1:], args.latente, f"AE_{clase}")
        # Entrada = salida esperada (X_c, X_c): el autoencoder aprende a copiar su clase.
        # Se repite el conjunto para que las clases pequeñas reciban tantas actualizaciones como las grandes.
        ds = tf.data.Dataset.from_tensor_slices((X_c, X_c)).shuffle(len(X_c), seed=SEMILLA).repeat().batch(16)
        h = ae.fit(ds, epochs=args.epocas, steps_per_epoch=PASOS_POR_EPOCA, verbose=0, shuffle=False,
                   validation_data=(X_pru[y_pru == c], X_pru[y_pru == c]))
        # Umbral binario: el error que no supera el 95% de las imágenes de entrenamiento de su clase
        e_ent = error_reconstruccion(ae, X_ent[y_ent == c])[0]
        umbrales.append(float(np.percentile(e_ent, 95)))
        print(f"  MSE entrenamiento = {h.history['loss'][-1]:.5f} | umbral binario = {umbrales[-1]:.5f}")
        autoencoders.append(ae)
        historias.append(h.history)
    graficar_entrenamiento(historias, clases, "1_entrenamiento_autoencoders.png")

    # ===== 2) Clasificar el 20% de prueba: gana el autoencoder con menor error =====
    # Error típico de cada autoencoder sobre TODAS las imágenes de entrenamiento (para el criterio normalizado)
    escala = clasificar(autoencoders, X_ent)[1].mean(axis=0) if args.criterio == "normalizado" else None
    pred, err_pru, punt_pru = clasificar(autoencoders, X_pru, escala)
    print(f"Criterio: {args.criterio}")
    print(f"\nExactitud (autoencoder más cercano): {np.mean(pred == y_pru):.1%}")

    idx = np.concatenate([np.where(y_pru == c)[0][:2] for c in range(len(clases))])
    graficar_entrada_vs_salida(X_pru[idx], y_pru[idx], autoencoders, clases, "2_entrada_vs_salida.png", escala)
    graficar_mapas_error(X_pru[idx], y_pru[idx], autoencoders, clases, "3_mapas_de_error.png")
    graficar_matriz_errores(punt_pru, y_pru, clases, "4_errores_prueba.png",
                            "MSE" if escala is None else "MSE / error típico del AE")
    graficar_matriz_confusion(y_pru, pred, clases, "5_matriz_confusion.png")
    binarios = graficar_binarios(err_pru, y_pru, umbrales, clases, "6_clasificadores_binarios.png")
    graficar_predicciones_prueba(X_pru, y_pru, pred, err_pru, clases, "7_predicciones_prueba.png")

    # ===== Reporte =====
    lineas = ["=== Clasificación final: autoencoder con menor error de reconstrucción ===",
              classification_report(y_pru, pred, labels=range(len(clases)), target_names=clases, zero_division=0),
              "=== Cada autoencoder como clasificador binario (clase vs. resto) ==="]
    for clase, umbral, auc, acc, cm in binarios:
        lineas.append(f"AE-{clase:<11} umbral={umbral:.5f}  AUC={auc:.2f}  exactitud={acc:.1%}  "
                      f"[VP={cm[0, 0]} FN={cm[0, 1]} FP={cm[1, 0]} VN={cm[1, 1]}]")
    with open(os.path.join(CARPETA_SALIDA, "reporte.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lineas) + "\n")
    print("\n" + "\n".join(lineas))
    print(f"\nListo. Revisa la carpeta '{CARPETA_SALIDA}/'.")


if __name__ == "__main__":
    main()
