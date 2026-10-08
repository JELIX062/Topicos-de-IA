"""
Clasificación de galaxias (Elliptical / Espiral / Lenticular) con:

  0) Un AUTOENCODER DETECTOR DE GALAXIAS: aprende a reconstruir galaxias (de
     cualquier clase). Si una imagen nueva no se parece a una galaxia, la
     reconstruye mal y se rechaza como "no es una galaxia".

  1) Un SISTEMA DE 6 AUTOENCODERS (dos por clase, en cadena):
       - 3 autoencoders de LIMPIEZA (de-noising), uno por clase: reciben la
         imagen y devuelven una versión limpia (sin ruido ni estrellas).
       - 3 autoencoders BINARIOS, uno por clase: reciben la imagen limpia, la
         reconstruyen y, con una neurona sigmoide sobre su código latente,
         responden "¿pertenece a mi clase?" (sí / no).
     Una imagen nueva pasa por las 3 cadenas (limpieza -> binario) y se asigna a
     la clase cuyo autoencoder binario da la MAYOR probabilidad de "sí".

  2) Una CNN clásica (Convolución + ReLU -> Pooling) x3 -> Flatten -> Densa ->
     Softmax, y se grafica cómo la imagen pasa por cada filtro / capa, al
     estilo del diagrama "Convolution Neural Network (CNN)".

División de datos: la prueba se calcula como el 20% de la clase MÁS PEQUEÑA
(18 elípticas -> 4 imágenes) y se toma ese MISMO número de cada clase; el
resto es entrenamiento.

Uso:
    python clasificador_galaxias.py                 # entrena y genera todo
    python clasificador_galaxias.py --imagen data/Espiral/NGC24.jpg
    python clasificador_galaxias.py --nuevas foto1.jpg carpeta_con_imagenes/

Todas las figuras se guardan en la carpeta  resultados/
"""

import argparse
import glob
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image, ImageOps
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.utils.class_weight import compute_class_weight

import keras
import tensorflow as tf
from keras import layers

SEMILLA = 42
PORC_PRUEBA = 0.20  # 20% de la clase más pequeña, mismo número de imágenes para cada clase
PASOS_POR_EPOCA = 30  # mismo número de actualizaciones para los autoencoders de todas las clases
CARPETA_SALIDA = "resultados"


# ---------------------------------------------------------------------------
# Datos
# ---------------------------------------------------------------------------
def cargar_imagen(ruta, tam):
    """Lee una imagen en escala de grises, recorta el centro y la normaliza."""
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


def aumentar(X, y):
    """Las galaxias no tienen 'arriba' ni 'abajo': se generan las 8 rotaciones/espejos."""
    Xs, ys = [], []
    for k in range(4):
        rot = np.rot90(X, k, axes=(1, 2))
        Xs += [rot, rot[:, :, ::-1]]
        ys += [y, y]
    return np.concatenate(Xs), np.concatenate(ys)


def dividir_prueba_igual(y, porc, semilla):
    """El % de prueba se calcula sobre la clase MÁS PEQUEÑA y se toma ese mismo número de cada clase.

    Con 18 elípticas: 20% de 18 = 3.6 -> 4 imágenes de prueba por clase (12 en total).
    """
    rng = np.random.default_rng(semilla)
    n_prueba = max(1, int(round(np.bincount(y).min() * porc)))
    idx_pru = np.concatenate([rng.choice(np.where(y == c)[0], n_prueba, replace=False) for c in np.unique(y)])
    idx_ent = np.setdiff1d(np.arange(len(y)), idx_pru)
    return idx_ent, np.sort(idx_pru), n_prueba


def ejemplos_no_galaxias(tam):
    """Imágenes que NO son galaxias, generadas aquí mismo, para probar el detector sin subir nada."""
    from matplotlib import cbook
    from PIL import ImageDraw
    imgs = {}
    for archivo, nombre in [("grace_hopper.jpg", "persona"), ("Minduka_Present_Blue_Pack.png", "regalo"),
                            ("logo2.png", "logotipo")]:
        try:  # fotos de ejemplo que vienen incluidas con matplotlib
            imgs[nombre] = Image.open(cbook.get_sample_data(archivo, asfileobj=False))
        except (OSError, ValueError):
            pass
    texto = Image.new("L", (300, 300), 255)
    dibujo = ImageDraw.Draw(texto)
    for i in range(8):
        dibujo.text((20, 20 + 32 * i), "Topicos de IA - galaxias " * 2, fill=0)
    imgs["texto"] = texto
    fig, eje = plt.subplots(figsize=(4, 3))
    eje.plot(np.random.default_rng(0).normal(size=50).cumsum())
    fig.canvas.draw()
    imgs["gráfica"] = Image.fromarray(np.asarray(fig.canvas.buffer_rgba()))
    plt.close(fig)
    r = np.random.default_rng(1)
    imgs["ruido"] = Image.fromarray((r.random((200, 200)) * 255).astype("uint8"))
    imgs["degradado"] = Image.fromarray(np.tile(np.linspace(0, 255, 200), (200, 1)).astype("uint8"))
    imgs["ajedrez"] = Image.fromarray(((np.indices((200, 200)) // 25).sum(0) % 2 * 255).astype("uint8"))
    imgs["negro"] = Image.new("L", (200, 200), 0)
    imgs["blanco"] = Image.new("L", (200, 200), 255)
    X = np.stack([np.asarray(ImageOps.fit(im.convert("L"), (tam, tam), Image.LANCZOS), dtype="float32")[..., None]
                  / 255.0 for im in imgs.values()])
    return X, list(imgs)


EXTENSIONES = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


def cargar_nuevas(rutas, tam):
    """Carga imágenes nuevas (archivos sueltos o carpetas) con el mismo preprocesamiento."""
    archivos = []
    for r in rutas:
        if os.path.isdir(r):
            archivos += sorted(f for f in glob.glob(os.path.join(r, "**", "*"), recursive=True)
                               if f.lower().endswith(EXTENSIONES))
        elif r.lower().endswith(EXTENSIONES):
            archivos.append(r)
    if not archivos:
        return np.empty((0, tam, tam, 1), "float32"), []
    return np.stack([cargar_imagen(f, tam) for f in archivos]), archivos


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def construir_ae_limpieza(forma, nombre):
    """Autoencoder de LIMPIEZA (de-noising): recibe una imagen con ruido y devuelve la imagen limpia."""
    entrada = keras.Input(forma, name="entrada")
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="enc_conv1")(entrada)
    x = layers.MaxPooling2D(name="enc_pool1")(x)                                  # 32x32
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="enc_conv2")(x)
    x = layers.MaxPooling2D(name="enc_pool2")(x)                                  # 16x16
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="enc_conv3")(x)
    codigo = layers.MaxPooling2D(name="codigo_latente")(x)                        # 8x8x64
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu", name="dec_up1")(codigo)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu", name="dec_up2")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu", name="dec_up3")(x)
    salida = layers.Conv2D(forma[-1], 3, padding="same", activation="sigmoid", name="imagen_limpia")(x)
    ae = keras.Model(entrada, salida, name=nombre)
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae


def construir_ae_binario(forma, latente, nombre):
    """Autoencoder BINARIO: reconstruye la imagen limpia pasando por un cuello de botella pequeño.

    Como solo aprende con imágenes de SU clase, reconstruye bien su clase (error pequeño = "sí")
    y mal las demás (error grande = "no").
    """
    entrada = keras.Input(forma, name="entrada")
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(entrada)
    x = layers.MaxPooling2D()(x)                                                  # 32x32
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)                                                  # 16x16
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)                                                  # 8x8
    x = layers.Flatten()(x)
    codigo = layers.Dense(latente, activation="relu", name="codigo_latente")(x)
    x = layers.Dense(8 * 8 * 32, activation="relu")(codigo)
    x = layers.Reshape((8, 8, 32))(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(16, 3, strides=2, padding="same", activation="relu")(x)
    salida = layers.Conv2D(forma[-1], 3, padding="same", activation="sigmoid", name="reconstruccion")(x)
    ae = keras.Model(entrada, salida, name=nombre)
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return ae


def entrenar_ae(ae, X_entrada, X_objetivo, epocas, validacion, ruido=0.0):
    """Entrena con el MISMO número de pasos para todas las clases (las pequeñas repiten más sus imágenes).

    Si ruido > 0 se agrega ruido gaussiano nuevo a la entrada en cada lote (autoencoder de limpieza).
    """
    ds = tf.data.Dataset.from_tensor_slices((X_entrada, X_objetivo))
    ds = ds.shuffle(len(X_entrada), seed=SEMILLA).repeat().batch(16)
    if ruido > 0:
        ds = ds.map(lambda a, b: (tf.clip_by_value(a + ruido * tf.random.normal(tf.shape(a)), 0.0, 1.0), b))
    return ae.fit(ds, epochs=epocas, steps_per_epoch=PASOS_POR_EPOCA, validation_data=validacion,
                  verbose=0, shuffle=False)


def error_relativo(ae, X):
    """Error de reconstrucción dividido entre la varianza (contraste) de la imagen: MSE / var.

    Así una imagen casi plana (toda negra, toda blanca, ruido...) no "pasa" como galaxia solo por
    tener poco que reconstruir.
    """
    rec = ae.predict(X, verbose=0, batch_size=64)
    mse = np.mean((X - rec) ** 2, axis=(1, 2, 3))
    return mse / (X.var(axis=(1, 2, 3)) + 1e-4), rec


def construir_ae_binario_si(forma, latente, nombre):
    """Autoencoder BINARIO de una clase: reconstruye la imagen limpia Y responde '¿es de mi clase?'.

    Tiene dos salidas: la reconstrucción (decoder) y una neurona sigmoide conectada al código latente
    que da la probabilidad de 'sí'. Aprender a reconstruir obliga al código latente a resumir la
    galaxia; la neurona sigmoide aprende qué de ese resumen distingue a su clase de las demás.
    """
    base = construir_ae_binario(forma, latente, nombre + "_base")
    codigo = base.get_layer("codigo_latente").output
    si = layers.Dense(1, activation="sigmoid", name="si")(layers.Dropout(0.4)(codigo))
    ae = keras.Model(base.input, [base.output, si], name=nombre)
    ae.compile(optimizer=keras.optimizers.Adam(1e-3), loss=["mse", "binary_crossentropy"], loss_weights=[1.0, 0.1])
    return ae


def entrenar_binario(ae, X_limpias, es_clase, epocas, validacion):
    """Entrena con el mismo número de imágenes 'sí' y 'no' (se repiten las de la clase pequeña)."""
    rng = np.random.default_rng(SEMILLA)
    pos, neg = np.where(es_clase)[0], np.where(~es_clase)[0]
    n = max(len(pos), len(neg))
    sel = np.concatenate([rng.choice(pos, n, replace=len(pos) < n), rng.choice(neg, n, replace=len(neg) < n)])
    t = es_clase[sel].astype("float32")
    return ae.fit(X_limpias[sel], [X_limpias[sel], t], epochs=epocas, batch_size=32, shuffle=True, verbose=0,
                  validation_data=validacion)


def pasar_por_autoencoders(limpiadores, binarios, X):
    """Cada imagen pasa por las 3 cadenas  AE-limpieza(c) -> AE-binario(c).

    Devuelve las imágenes limpias, las reconstrucciones y P(sí) (n_imagenes, n_clases) de cada
    autoencoder binario.
    """
    limpias = [ae.predict(X, verbose=0, batch_size=64) for ae in limpiadores]
    salidas = [ae.predict(l, verbose=0, batch_size=64) for ae, l in zip(binarios, limpias)]
    recs = [s_[0] for s_ in salidas]
    probs = np.stack([s_[1][:, 0] for s_ in salidas], axis=1)
    return limpias, recs, probs


def construir_cnn(forma, n_clases):
    """Misma estructura que el diagrama: (Conv+ReLU -> Pool) x3 -> Flatten -> FC -> Softmax."""
    entrada = keras.Input(forma, name="entrada")
    x = entrada
    for i, filtros in enumerate([16, 32, 64], start=1):
        x = layers.Conv2D(filtros, 3, padding="same", activation="relu", name=f"conv{i}")(x)
        x = layers.MaxPooling2D(name=f"pool{i}")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu", name="densa")(x)
    x = layers.Dropout(0.3)(x)
    salida = layers.Dense(n_clases, activation="softmax", name="softmax")(x)
    modelo = keras.Model(entrada, salida, name="cnn")
    modelo.compile(optimizer=keras.optimizers.Adam(5e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


def agregar_ruido(X, factor, rng):
    return np.clip(X + factor * rng.normal(size=X.shape).astype("float32"), 0.0, 1.0)


# ---------------------------------------------------------------------------
# Gráficas generales
# ---------------------------------------------------------------------------
def guardar(fig, nombre):
    ruta = os.path.join(CARPETA_SALIDA, nombre)
    fig.savefig(ruta, dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  -> {ruta}")


def graficar_historial(historias, titulo, nombre):
    fig, ejes = plt.subplots(1, len(historias[0][1]) // 2, figsize=(12, 4))
    ejes = np.atleast_1d(ejes)
    metricas = [m for m in historias[0][1] if not m.startswith("val_")]
    for eje, m in zip(ejes, metricas):
        for etiqueta, h in historias:
            eje.plot(h[m], label=f"{etiqueta} entrenamiento")
            eje.plot(h["val_" + m], "--", label=f"{etiqueta} validación")
        eje.set_title(m)
        eje.set_xlabel("época")
        eje.grid(alpha=0.3)
        eje.legend(fontsize=8)
    fig.suptitle(titulo)
    guardar(fig, nombre)


def graficar_matriz_confusion(y_real, y_pred, clases, titulo, nombre):
    cm = confusion_matrix(y_real, y_pred, labels=range(len(clases)))
    fig, eje = plt.subplots(figsize=(5, 4.5))
    im = eje.imshow(cm, cmap="Blues")
    for i in range(len(clases)):
        for j in range(len(clases)):
            eje.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    eje.set_xticks(range(len(clases)), clases, rotation=20)
    eje.set_yticks(range(len(clases)), clases)
    eje.set_xlabel("Predicción")
    eje.set_ylabel("Real")
    acc = (cm.trace() / cm.sum()) * 100
    eje.set_title(f"{titulo}\nexactitud = {acc:.1f}%")
    fig.colorbar(im, ax=eje, fraction=0.046)
    guardar(fig, nombre)


def graficar_predicciones_prueba(X, y, pred_ae, pred_cnn, clases, nombre, columnas=10):
    """Todas las imágenes del 20% de prueba con la predicción de ambos modelos (verde = acierto)."""
    n = len(X)
    filas = int(np.ceil(n / columnas))
    fig, ejes = plt.subplots(filas, columnas, figsize=(1.9 * columnas, 2.7 * filas), squeeze=False)
    for k, eje in enumerate(ejes.flat):
        eje.axis("off")
        if k >= n:
            continue
        eje.imshow(X[k, ..., 0], cmap="gray")
        p_ae, p_cnn = int(pred_ae[k]), int(pred_cnn[k])
        eje.set_title(f"Real: {clases[y[k]]}", fontsize=8)
        eje.text(0.5, -0.04, f"6 AE: {clases[p_ae]}", transform=eje.transAxes, ha="center", va="top", fontsize=7.5,
                 color="green" if p_ae == y[k] else "red")
        eje.text(0.5, -0.17, f"CNN: {clases[p_cnn]}", transform=eje.transAxes, ha="center", va="top",
                 fontsize=7.5, color="green" if p_cnn == y[k] else "red")
    acc_ae = np.mean(pred_ae == y) * 100
    acc_cnn = np.mean(pred_cnn == y) * 100
    fig.suptitle(f"Conjunto de prueba ({n} imágenes no vistas en el entrenamiento)\n"
                 f"6 autoencoders: {acc_ae:.1f}%   |   CNN: {acc_cnn:.1f}%   (verde = acierto, rojo = error)",
                 fontsize=13)
    fig.subplots_adjust(hspace=0.65, wspace=0.08, top=1 - 0.7 / filas, bottom=0.25 / filas)
    guardar(fig, nombre)

# ---------------------------------------------------------------------------
# Autoencoders: limpieza y clasificación binaria
# ---------------------------------------------------------------------------
COLORES = ["#1f77b4", "#ff7f0e", "#2ca02c"]


def nombre_real(c, clases):
    return clases[c] if c >= 0 else "desconocida"


def color_resultado(pred, real):
    """Verde = acierto, rojo = error, azul = imagen nueva sin clase conocida."""
    if real < 0:
        return "#1f4fd1"
    return "green" if pred == real else "red"


def graficar_detector(err_entrenamiento, grupos, umbral, nombre):
    """Histograma del error relativo del detector: galaxias de entrenamiento vs. otras imágenes."""
    todos = np.concatenate([err_entrenamiento] + [g[1] for g in grupos])
    bins = np.logspace(np.log10(todos.min() * 0.9), np.log10(todos.max() * 1.1), 45)
    fig, eje = plt.subplots(figsize=(11, 4.3))
    eje.hist(err_entrenamiento, bins=bins, color="#1f77b4", alpha=0.7, label="galaxias de entrenamiento")
    for etiqueta, e, color in grupos:
        eje.hist(e, bins=bins, color=color, alpha=0.75, label=etiqueta)
    eje.axvline(umbral, color="red", ls="--", lw=2, label=f"umbral = {umbral:.3f}")
    eje.axvspan(bins[0], umbral, color="green", alpha=0.06)
    eje.axvspan(umbral, bins[-1], color="red", alpha=0.06)
    eje.text(umbral / 1.15, eje.get_ylim()[1] * 0.92, "← GALAXIA", ha="right", color="green", weight="bold")
    eje.text(umbral * 1.15, eje.get_ylim()[1] * 0.92, "NO ES GALAXIA →", ha="left", color="red", weight="bold")
    eje.set_xscale("log")
    eje.set_xlabel("error relativo de reconstrucción (MSE / varianza de la imagen), escala logarítmica")
    eje.set_ylabel("número de imágenes")
    eje.legend(fontsize=9, loc="center right")
    eje.set_title("Etapa 0 — Autoencoder DETECTOR DE GALAXIAS: si reconstruye bien la imagen, es una galaxia")
    fig.tight_layout()
    guardar(fig, nombre)


def graficar_deteccion(X, nombres, detector, umbral, nombre, columnas=8):
    """Entrada (arriba), reconstrucción del detector (abajo) y veredicto de cada imagen."""
    err, rec = error_relativo(detector, X)
    n = len(X)
    columnas = min(columnas, n)
    filas = int(np.ceil(n / columnas))
    fig, ejes = plt.subplots(2 * filas, columnas, figsize=(2.15 * max(columnas, 2), 4.7 * filas), squeeze=False)
    for eje in ejes.flat:
        eje.axis("off")
    for k in range(n):
        f, c = divmod(k, columnas)
        es = err[k] <= umbral
        color = "green" if es else "red"
        ejes[2 * f, c].imshow(X[k, ..., 0], cmap="gray", vmin=0, vmax=1)
        ejes[2 * f, c].set_title(str(nombres[k])[:20], fontsize=8)
        ejes[2 * f + 1, c].imshow(rec[k, ..., 0], cmap="gray", vmin=0, vmax=1)
        ejes[2 * f + 1, c].set_title(f"error = {err[k]:.2f}\n{'GALAXIA' if es else 'NO ES GALAXIA'}", fontsize=9,
                                     color=color, weight="bold")
        for eje in (ejes[2 * f, c], ejes[2 * f + 1, c]):
            eje.add_patch(Rectangle((0, 0), 1, 1, transform=eje.transAxes, fill=False, ec=color, lw=3))
    fig.suptitle(f"Detector de galaxias: imagen (arriba) y lo que reconstruye el autoencoder detector (abajo)\n"
                 f"si el error relativo ≤ {umbral:.3f} → es una galaxia", fontsize=12)
    fig.tight_layout()
    guardar(fig, nombre)
    return err <= umbral, err


def graficar_entrenamiento_aes(historias, clases, titulo, nombre):
    """Curva de error de cada uno de los 3 autoencoders (entrenamiento vs. prueba de su clase)."""
    fig, ejes = plt.subplots(1, len(clases), figsize=(5 * len(clases), 3.8), sharey=True)
    for eje, h, clase, color in zip(ejes, historias, clases, COLORES):
        eje.plot(h["loss"], color=color, label="entrenamiento")
        eje.plot(h["val_loss"], "--", color=color, alpha=0.7, label=f"prueba ({clase})")
        eje.set_title(clase)
        eje.set_xlabel("época")
        eje.set_yscale("log")
        eje.grid(alpha=0.3)
        eje.legend(fontsize=8)
    ejes[0].set_ylabel("error (MSE)")
    fig.suptitle(titulo, fontsize=13)
    fig.tight_layout()
    guardar(fig, nombre)


def graficar_limpieza(X, X_ruido, y, limpiadores, clases, nombre):
    """ANTES (original / con ruido) y DESPUÉS (imagen limpia que entrega cada autoencoder de limpieza)."""
    k = len(clases)
    limpias = [ae.predict(X_ruido, verbose=0) for ae in limpiadores]
    fig, ejes = plt.subplots(len(X), k + 2, figsize=(2.4 * (k + 2), 2.55 * len(X)), squeeze=False)
    for i in range(len(X)):
        ejes[i, 0].imshow(X[i, ..., 0], cmap="gray", vmin=0, vmax=1)
        ejes[i, 0].set_title(f"ANTES: original\n({nombre_real(y[i], clases)})", fontsize=9)
        ejes[i, 1].imshow(X_ruido[i, ..., 0], cmap="gray", vmin=0, vmax=1)
        ejes[i, 1].set_title("ANTES: con ruido", fontsize=9)
        for j in range(k):
            ejes[i, j + 2].imshow(limpias[j][i, ..., 0], cmap="gray", vmin=0, vmax=1)
            ejes[i, j + 2].set_title(f"DESPUÉS: limpia\nAE-limpieza {clases[j]}", fontsize=9,
                                     weight="bold" if j == y[i] else "normal")
        for eje in ejes[i]:
            eje.set_xticks([])
            eje.set_yticks([])
    fig.suptitle("Etapa 1 — Autoencoders de LIMPIEZA: la imagen antes y después (en negritas, el de su clase)",
                 fontsize=13)
    fig.tight_layout()
    guardar(fig, nombre)


def graficar_cadena(X, y, limpiadores, binarios, clases, nombre):
    """Recorrido completo: entrada -> limpia(c) -> AE-binario(c) (reconstrucción + 'sí'/'no') + decisión."""
    k = len(clases)
    limpias, recs, probs = pasar_por_autoencoders(limpiadores, binarios, X)
    fig, ejes = plt.subplots(len(X), 2 * k + 2, figsize=(2.05 * (2 * k + 2), 2.75 * len(X)), squeeze=False,
                             gridspec_kw={"width_ratios": [1] * (2 * k + 1) + [1.4]})
    for i in range(len(X)):
        gana = int(probs[i].argmax())
        ejes[i, 0].imshow(X[i, ..., 0], cmap="gray", vmin=0, vmax=1)
        ejes[i, 0].set_title(f"ENTRADA\nreal: {nombre_real(y[i], clases)}", fontsize=8.5)
        for j in range(k):
            e_l, e_b = ejes[i, 1 + 2 * j], ejes[i, 2 + 2 * j]
            e_l.imshow(limpias[j][i, ..., 0], cmap="gray", vmin=0, vmax=1)
            e_l.set_title(f"limpia\n(AE-limpieza {clases[j]})", fontsize=8, color=COLORES[j])
            e_b.imshow(recs[j][i, ..., 0], cmap="gray", vmin=0, vmax=1)
            e_b.set_title(f"AE-binario {clases[j]}\nP(sí)={probs[i, j]:.2f}  {'SÍ' if probs[i, j] >= 0.5 else 'NO'}",
                          fontsize=8, color=COLORES[j], weight="bold" if j == gana else "normal")
            if j == gana:
                for e in (e_l, e_b):
                    e.add_patch(Rectangle((0, 0), 1, 1, transform=e.transAxes, fill=False,
                                          ec=color_resultado(gana, y[i]), lw=4))
        for eje in ejes[i, :-1]:
            eje.set_xticks([])
            eje.set_yticks([])
        barras = ejes[i, -1]
        barras.barh(range(k), probs[i], color=COLORES[:k])
        barras.axvline(0.5, color="red", lw=1.5)
        barras.set_xlim(0, 1)
        barras.set_yticks(range(k), clases, fontsize=8)
        barras.invert_yaxis()
        barras.tick_params(axis="x", labelsize=7)
        barras.set_title(f"Predicción: {clases[gana]}", fontsize=10, weight="bold",
                         color=color_resultado(gana, y[i]))
    fig.suptitle("Cada imagen pasa por las 3 cadenas  AE-limpieza → AE-binario.  Cada AE-binario reconstruye la imagen "
                 "limpia y responde '¿es de mi clase?' (P(sí))\nGana la clase con MAYOR P(sí)  (línea roja = 0.5; "
                 "marco verde = acierto, rojo = error, azul = imagen nueva)", fontsize=12)
    fig.tight_layout()
    guardar(fig, nombre)


def graficar_matriz_probabilidades(probs, y, clases, nombre):
    """P(sí) de cada imagen de prueba en cada autoencoder binario (filas ordenadas por clase real)."""
    orden = np.argsort(y, kind="stable")
    P = probs[orden]
    fig, eje = plt.subplots(figsize=(6, 0.32 * len(P) + 1.8))
    im = eje.imshow(P, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    for i, fila in enumerate(P):
        j = fila.argmax()
        eje.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                ec="lime" if j == y[orden][i] else "red", lw=2))
    for b in np.cumsum([np.sum(y == c) for c in range(len(clases))])[:-1]:
        eje.axhline(b - 0.5, color="white", lw=2)
    centros = [np.mean(np.where(y[orden] == c)[0]) for c in range(len(clases))]
    eje.set_yticks(centros, [f"real: {c}" for c in clases])
    eje.set_xticks(range(len(clases)), [f"AE-binario\n{c}" for c in clases])
    eje.set_title("P(sí) de cada imagen de prueba en cada autoencoder binario\n"
                  "cuadro = clase elegida (verde = correcto, rojo = incorrecto)", fontsize=11)
    fig.colorbar(im, ax=eje, label="P(sí es de mi clase)")
    fig.tight_layout()
    guardar(fig, nombre)


def graficar_binarios(probs, y, clases, nombre):
    """Cada autoencoder binario por separado: '¿pertenece a mi clase?' (P(sí) >= 0.5)."""
    k = len(clases)
    fig, ejes = plt.subplots(2, k, figsize=(5.2 * k, 8), gridspec_kw={"height_ratios": [1.3, 1]})
    resumen = []
    bins = np.linspace(0, 1, 21)
    for c in range(k):
        p = probs[:, c]
        eje = ejes[0, c]
        eje.hist(p[y != c], bins=bins, alpha=0.6, color="gray", label="otras clases")
        eje.hist(p[y == c], bins=bins, alpha=0.8, color=COLORES[c], label=clases[c])
        eje.axvline(0.5, color="red", ls="--", label="umbral = 0.5")
        auc = roc_auc_score(y == c, p)
        eje.set_title(f"AE-binario {clases[c]}  (AUC = {auc:.2f})")
        eje.set_xlabel("P(sí es de mi clase)")
        eje.legend(fontsize=8)

        cm = confusion_matrix((y == c).astype(int), (p >= 0.5).astype(int), labels=[1, 0])
        eje = ejes[1, c]
        eje.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                eje.text(j, i, cm[i, j], ha="center", va="center", fontsize=14,
                         color="white" if cm[i, j] > cm.max() / 2 else "black")
        etiquetas = [f"Sí es {clases[c]}", "No es"]
        eje.set_xticks([0, 1], etiquetas)
        eje.set_yticks([0, 1], etiquetas)
        eje.set_xlabel("Predicción (P(sí) ≥ 0.5)")
        eje.set_ylabel("Real")
        acc = (cm[0, 0] + cm[1, 1]) / cm.sum()
        eje.set_title(f"Binario {clases[c]} vs. resto: exactitud = {acc:.1%}")
        resumen.append((clases[c], 0.5, auc, acc, cm))
    fig.suptitle("Etapa 2 — Cada AUTOENCODER BINARIO por separado: '¿la imagen pertenece a mi clase?'", fontsize=13)
    fig.tight_layout()
    guardar(fig, nombre)
    return resumen


# ---------------------------------------------------------------------------
# CNN: recorrido por cada filtro
# ---------------------------------------------------------------------------
def obtener_activaciones(modelo, nombres, imagen):
    sub = keras.Model(modelo.input, [modelo.get_layer(n).output for n in nombres])
    salidas = sub.predict(imagen[None], verbose=0)
    return {n: s[0] for n, s in zip(nombres, salidas)}


def graficar_filtros_conv1(modelo, activ_conv1, imagen, nombre):
    """Cada kernel 3x3 de la primera capa y el mapa de características que produce."""
    pesos = modelo.get_layer("conv1").get_weights()[0][..., 0, :]  # (3,3,filtros)
    n = pesos.shape[-1]
    fig, ejes = plt.subplots(4, n // 2 + 1, figsize=(2 * (n // 2 + 1), 8.5),
                             gridspec_kw={"height_ratios": [1, 2, 1, 2]})
    for eje in ejes.flat:
        eje.axis("off")
    for fila in (1, 3):
        ejes[fila, 0].imshow(imagen[..., 0], cmap="gray")
        ejes[fila, 0].set_title("Entrada", fontsize=9)
    for k in range(n):
        fila, col = (0, k + 1) if k < n // 2 else (2, k - n // 2 + 1)
        ejes[fila, col].imshow(pesos[..., k], cmap="coolwarm")
        ejes[fila, col].set_title(f"Kernel {k + 1}", fontsize=9)
        ejes[fila + 1, col].imshow(activ_conv1[..., k], cmap="viridis")
        ejes[fila + 1, col].set_title(f"Mapa {k + 1}", fontsize=9)
    fig.suptitle("Capa conv1: kernel 3x3 aprendido (arriba) -> mapa de características tras Conv + ReLU (abajo)",
                 fontsize=13)
    fig.tight_layout()
    guardar(fig, nombre)


def graficar_mapas_por_capa(activs, imagen, nombre, max_mapas=16):
    """Una fila por capa: así se transforma la imagen al pasar por cada filtro."""
    capas = list(activs)
    fig, ejes = plt.subplots(len(capas), max_mapas + 1, figsize=(1.35 * (max_mapas + 1), 1.55 * len(capas)))
    for i, capa in enumerate(capas):
        a = activs[capa]
        orden = np.argsort(a.mean(axis=(0, 1)))[::-1][:max_mapas]
        ejes[i, 0].imshow(imagen[..., 0], cmap="gray")
        ejes[i, 0].set_ylabel(f"{capa}\n{a.shape[0]}x{a.shape[1]}x{a.shape[2]}", fontsize=9)
        for j in range(max_mapas + 1):
            ejes[i, j].set_xticks([])
            ejes[i, j].set_yticks([])
        for j, k in enumerate(orden, start=1):
            ejes[i, j].imshow(a[..., k], cmap="viridis")
            ejes[i, j].set_title(f"f{k}", fontsize=7)
    fig.suptitle("Mapas de características de cada capa (los 16 filtros más activos)", fontsize=13)
    fig.tight_layout()
    guardar(fig, nombre)


def _normalizar(m):
    m = m - m.min()
    return m / (m.max() + 1e-8)


def graficar_diagrama_cnn(modelo, imagen, clases, clase_real, nombre):
    """Reproduce el diagrama 'Convolution Neural Network' usando las activaciones reales."""
    capas_mapas = ["conv1", "pool1", "conv2", "pool2", "conv3", "pool3"]
    activs = obtener_activaciones(modelo, capas_mapas + ["flatten", "densa", "softmax"], imagen)

    fig = plt.figure(figsize=(24, 10.5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 240)
    ax.set_ylim(-8, 100)
    ax.axis("off")
    ax.text(120, 95, "Red Neuronal Convolucional (CNN)", ha="center", fontsize=24, weight="bold")

    centro_y = 52
    # ---------------- Entrada ----------------
    lado_in = 24
    x_in, y_in = 3, centro_y - lado_in / 2
    ax.imshow(imagen[..., 0], cmap="gray", extent=[x_in, x_in + lado_in, y_in, y_in + lado_in], zorder=2)
    ax.add_patch(Rectangle((x_in, y_in), lado_in, lado_in, fill=False, lw=1.5, zorder=3))
    ax.text(x_in + lado_in / 2, y_in + lado_in + 4, "Entrada", ha="center", fontsize=17)
    ax.text(x_in + lado_in / 2, y_in + lado_in + 1, f"{imagen.shape[0]}x{imagen.shape[1]}", ha="center", fontsize=10)
    k_in = (x_in + lado_in * 0.55, y_in + lado_in * 0.35, 3.5)
    ax.add_patch(Rectangle(k_in[:2], k_in[2], k_in[2], fill=False, ec="yellow", lw=2.5, zorder=4))
    ax.add_patch(FancyArrowPatch((k_in[0] + 1.7, k_in[1]), (k_in[0] + 1.7, y_in - 7),
                                 arrowstyle="-|>", mutation_scale=15, color="black", zorder=4))
    ax.text(k_in[0] + 1.7, y_in - 10, "Kernel", ha="center", fontsize=14)
    origen = (k_in[0] + k_in[2], k_in[1] + k_in[2] / 2)

    # ---------------- Pilas de mapas de características ----------------
    lados = {64: 20, 32: 15, 16: 11, 8: 8}
    x = 33
    posiciones_x = []
    for idx, capa in enumerate(capas_mapas):
        a = activs[capa]
        h, n_filtros = a.shape[0], a.shape[-1]
        lado = lados.get(h, max(6, h / 4))
        n_mostrar = min(8, n_filtros)
        desp = 1.1 if idx < 4 else 0.9
        orden = np.argsort(a.mean(axis=(0, 1)))[:: -1][:n_mostrar]  # el más activo queda al frente
        y0 = centro_y - lado / 2
        for prof, k in enumerate(orden[::-1]):  # del fondo hacia el frente
            off = (n_mostrar - 1 - prof) * desp
            ext = [x + off * 0.6, x + off * 0.6 + lado, y0 + off, y0 + off + lado]
            ax.imshow(_normalizar(a[..., k]), cmap="viridis", extent=ext, zorder=5 + prof)
            ax.add_patch(Rectangle((ext[0], ext[2]), lado, lado, fill=False, lw=0.8, ec="#333", zorder=5 + prof))
        # Kernel amarillo en el mapa del frente y líneas punteadas desde la capa anterior
        destino = (x + lado * 0.5, y0 + lado * 0.5)
        ax.plot([origen[0], destino[0]], [origen[1] + 1.5, destino[1]], "k--", lw=0.9, zorder=30)
        ax.plot([origen[0], destino[0]], [origen[1] - 1.5, destino[1]], "k--", lw=0.9, zorder=30)
        kl = max(1.6, lado * 0.17)
        kx, ky = x + lado * 0.55, y0 + lado * 0.25
        ax.add_patch(Rectangle((kx, ky), kl, kl, fill=False, ec="yellow", lw=2.2, zorder=31))
        origen = (kx + kl, ky + kl / 2)

        es_conv = capa.startswith("conv")
        etiqueta = "Convolución\n+\nReLU" if es_conv else "Pooling\n(máx 2x2)"
        y_txt = y0 - 4 if es_conv else y0 + lado + n_mostrar * desp + 7
        ax.text(x + lado / 2, y_txt, etiqueta, ha="center", va="top" if es_conv else "bottom", fontsize=13)
        ax.text(x + lado / 2, y0 - 16.5 if es_conv else y0 + lado + n_mostrar * desp + 2.5,
                f"{capa}: {n_filtros} mapas {h}x{a.shape[1]}", ha="center",
                va="top" if es_conv else "bottom", fontsize=9, color="#555")
        posiciones_x.append(x)
        x += lado + n_mostrar * desp * 0.6 + 5

    # ---------------- Flatten ----------------
    plano = activs["flatten"]
    n_celdas = 16
    valores = _normalizar(plano[np.argsort(plano)[::-1][:n_celdas]])
    x_f, alto_c, ancho_c = x + 6, 3.6, 3.2
    y_f0 = centro_y - n_celdas * alto_c / 2
    cmap = plt.get_cmap("viridis")
    for i, v in enumerate(valores):
        ax.add_patch(Rectangle((x_f, y_f0 + i * alto_c), ancho_c, alto_c, fc=cmap(v), ec="#5a4630", lw=1, zorder=5))
    ax.plot([origen[0], x_f], [origen[1], y_f0 + n_celdas * alto_c], "k--", lw=0.9)
    ax.plot([origen[0], x_f], [origen[1], y_f0], "k--", lw=0.9)
    ax.text(x_f + ancho_c / 2, y_f0 - 3, f"Capa\nFlatten\n({plano.size} valores)", ha="center", va="top", fontsize=12)

    # ---------------- Capas totalmente conectadas ----------------
    def nodos(xc, valores, color_base):
        n = len(valores)
        sep = min(5.5, 64 / max(n, 1))
        ys = centro_y + (np.arange(n) - (n - 1) / 2) * sep
        for yy, v in zip(ys, valores):
            ax.add_patch(Circle((xc, yy), 1.4, fc=plt.get_cmap(color_base)(0.35 + 0.65 * v), ec="#1d3557", zorder=6))
        return ys

    ys_flat = y_f0 + (np.arange(n_celdas) + 0.5) * alto_c
    x_d = x_f + 22
    densa = activs["densa"]
    top_d = np.argsort(densa)[::-1][:12]
    ys_d = nodos(x_d, _normalizar(densa[top_d]), "Blues")
    for y1 in ys_flat:
        for y2 in ys_d:
            ax.plot([x_f + ancho_c, x_d - 1.4], [y1, y2], color="black", lw=0.3, alpha=0.5, zorder=4)

    probs = activs["softmax"]
    x_s = x_d + 18
    ys_s = nodos(x_s, probs, "Blues")
    for y1 in ys_d:
        for y2 in ys_s:
            ax.plot([x_d + 1.4, x_s - 1.4], [y1, y2], color="black", lw=0.5, alpha=0.6, zorder=4)
    ax.text((x_d + x_s) / 2, y_f0 - 3, f"Capa totalmente\nconectada\n({densa.size} neuronas, se\nmuestran 12)",
            ha="center", va="top", fontsize=12)

    # ---------------- Salida softmax ----------------
    x_o = x_s + 6
    ancho_o, alto_o = 12, (ys_s.max() - ys_s.min()) + 10
    ax.add_patch(Rectangle((x_o, ys_s.min() - 5), ancho_o, alto_o, fc="#f2b880", ec="#b5651d", lw=1.5, zorder=3,
                           joinstyle="round"))
    pred = int(np.argmax(probs))
    for yy, p, clase, i in zip(ys_s, probs, clases, range(len(clases))):
        ax.text(x_o + 1, yy + 0.8, f"{p:.2f}", fontsize=14, color="white", weight="bold", zorder=7, va="center")
        ax.plot([x_s + 1.4, x_o + ancho_o + 1], [yy, yy], "k--", lw=0.9, zorder=6)
        ax.text(x_o + ancho_o + 1.5, yy, clase, fontsize=15, va="center",
                weight="bold" if i == pred else "normal", color="green" if i == pred else "black")
    ax.text(x_o + ancho_o / 2, ys_s.max() + 10, "Salida", ha="center", fontsize=18)
    ax.text(x_o + ancho_o / 2, ys_s.min() - 7, "Función de\nactivación\nSoftMax", ha="center", va="top", fontsize=12)

    # ---------------- Llaves inferiores ----------------
    def llave(x0, x1, texto):
        ax.plot([x0, x0, x1, x1], [2, 0, 0, 2], color="#555", lw=1)
        ax.text((x0 + x1) / 2, -3, texto, ha="center", va="top", fontsize=14)

    llave(2, x_f + ancho_c + 2, "Extracción de características")
    llave(x_f + ancho_c + 5, x_s + 3, "Clasificación")
    llave(x_s + 5, min(x_o + ancho_o + 14, 239), "Distribución\nprobabilística")
    ax.annotate("", xy=(posiciones_x[-1] + 8, 12), xytext=(posiciones_x[0], 12),
                arrowprops=dict(arrowstyle="<->", color="black"))
    ax.text((posiciones_x[0] + posiciones_x[-1] + 8) / 2, 13, "Mapas de características", ha="center", fontsize=14,
            bbox=dict(fc="white", ec="none"))

    if clase_real < 0:  # imagen externa sin etiqueta
        texto, color = f"Predicción: {clases[pred]} ({probs[pred]:.0%})", "black"
    else:
        resultado = "CORRECTO" if pred == clase_real else "INCORRECTO"
        texto = f"Clase real: {clases[clase_real]}   |   Predicción: {clases[pred]} ({probs[pred]:.0%})   [{resultado}]"
        color = "green" if pred == clase_real else "red"
    ax.text(120, 89, texto, ha="center", fontsize=14, color=color)
    ax.set_xlim(0, max(240, x_o + ancho_o + 16))
    guardar(fig, nombre)
    return activs


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datos", default="data", help="carpeta con una subcarpeta por clase")
    p.add_argument("--tam", type=int, default=64, help="tamaño (px) al que se redimensionan las imágenes")
    p.add_argument("--epocas-detector", type=int, default=20)
    p.add_argument("--epocas-limpieza", type=int, default=20)
    p.add_argument("--epocas-binario", type=int, default=25)
    p.add_argument("--epocas-cnn", type=int, default=40)
    p.add_argument("--latente", type=int, default=32, help="cuello de botella de los autoencoders binarios")
    p.add_argument("--ruido", type=float, default=0.15, help="ruido gaussiano para entrenar la limpieza")
    p.add_argument("--imagen", default=None, help="imagen a visualizar en el diagrama de la CNN")
    p.add_argument("--nuevas", nargs="*", default=[], help="imágenes o carpetas NUEVAS a clasificar al final")
    args = p.parse_args()

    keras.utils.set_random_seed(SEMILLA)
    rng = np.random.default_rng(SEMILLA)
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    print("Cargando imágenes...")
    X, y, rutas, clases = cargar_datos(args.datos, args.tam)
    idx_ent, idx_pru, n_prueba = dividir_prueba_igual(y, PORC_PRUEBA, SEMILLA)
    X_ent, y_ent, r_ent = X[idx_ent], y[idx_ent], rutas[idx_ent]
    X_pru, y_pru, r_pru = X[idx_pru], y[idx_pru], rutas[idx_pru]
    print(f"  Prueba: {PORC_PRUEBA:.0%} de la clase más pequeña ({np.bincount(y).min()}) = {n_prueba} por clase")
    for c, clase in enumerate(clases):
        print(f"  {clase:<11} entrenamiento: {np.sum(y_ent == c):3d}   prueba: {np.sum(y_pru == c)}")
    forma = X.shape[1:]
    idx = np.concatenate([np.where(y_pru == c)[0][:2] for c in range(len(clases))])  # ejemplos para graficar

    X_ent_a, y_ent_a = aumentar(X_ent, y_ent)  # 8 rotaciones/espejos por imagen

    # ================= 0) AUTOENCODER DETECTOR DE GALAXIAS =================
    print("\n[0/3] Entrenando el autoencoder DETECTOR DE GALAXIAS (con todas las galaxias de entrenamiento)...")
    detector = construir_ae_binario(forma, args.latente, "AE_detector_galaxias")
    entrenar_ae(detector, X_ent_a, X_ent_a, args.epocas_detector, (X_pru, X_pru))
    err_det_ent = error_relativo(detector, X_ent)[0]
    umbral_galaxia = float(np.percentile(err_det_ent, 99))  # el 99% de las galaxias de entrenamiento queda debajo
    X_no, nombres_no = ejemplos_no_galaxias(args.tam)
    err_det_pru = error_relativo(detector, X_pru)[0]
    err_det_no = error_relativo(detector, X_no)[0]
    print(f"  umbral = {umbral_galaxia:.3f} | galaxias de prueba aceptadas: {np.mean(err_det_pru <= umbral_galaxia):.0%}"
          f" | imágenes que NO son galaxias rechazadas: {np.mean(err_det_no > umbral_galaxia):.0%}")
    graficar_detector(err_det_ent, [("galaxias de prueba", err_det_pru, "#2ca02c"),
                                    ("imágenes que NO son galaxias", err_det_no, "#d62728")],
                      umbral_galaxia, "0_detector_histograma.png")
    graficar_deteccion(np.concatenate([X_pru[idx[::2]], X_no]),
                       [os.path.basename(r) for r in r_pru[idx[::2]]] + nombres_no,
                       detector, umbral_galaxia, "0_detector_ejemplos.png")

    # ================= 1) AUTOENCODERS DE LIMPIEZA (uno por clase) =================
    print("\n[1/3] Entrenando los 3 autoencoders de LIMPIEZA...")
    limpiadores, h_limpieza = [], []
    for c, clase in enumerate(clases):
        X_c = aumentar(X_ent[y_ent == c], y_ent[y_ent == c])[0]
        X_v = X_pru[y_pru == c]
        ae = construir_ae_limpieza(forma, f"AE_limpieza_{clase}")
        # Entrada con ruido -> salida esperada: la imagen original limpia
        h = entrenar_ae(ae, X_c, X_c, args.epocas_limpieza, (agregar_ruido(X_v, args.ruido, rng), X_v), args.ruido)
        print(f"  AE-limpieza {clase:<11} ({len(X_c)} imágenes con aumento)  MSE = {h.history['loss'][-1]:.5f}")
        limpiadores.append(ae)
        h_limpieza.append(h.history)
    graficar_entrenamiento_aes(h_limpieza, clases, "Etapa 1 — Autoencoders de LIMPIEZA (entrada con ruido → "
                               "imagen limpia)", "1_ae_limpieza_entrenamiento.png")
    graficar_limpieza(X_pru[idx], agregar_ruido(X_pru[idx], args.ruido, rng), y_pru[idx], limpiadores, clases,
                      "2_ae_limpieza_antes_despues.png")

    # ================= 2) AUTOENCODERS BINARIOS (uno por clase) =================
    print("\n[2/3] Entrenando los 3 autoencoders BINARIOS con las imágenes ya limpias...")
    binarios, h_binario = [], []
    for c, clase in enumerate(clases):
        # Cada AE-binario ve TODAS las clases (limpias por su AE-limpieza): las de su clase son "sí", las demás "no"
        X_l = limpiadores[c].predict(X_ent_a, verbose=0)
        X_v = limpiadores[c].predict(X_pru, verbose=0)
        ae = construir_ae_binario_si(forma, args.latente, f"AE_binario_{clase}")
        h = entrenar_binario(ae, X_l, y_ent_a == c, args.epocas_binario,
                             (X_v, [X_v, (y_pru == c).astype("float32")]))
        binarios.append(ae)
        h_binario.append(h.history)
        print(f"  AE-binario  {clase:<11} pérdida = {h.history['loss'][-1]:.4f}")
    graficar_entrenamiento_aes(h_binario, clases, "Etapa 2 — Autoencoders BINARIOS (reconstrucción + "
                               "'¿es de mi clase?')", "3_ae_binario_entrenamiento.png")

    # ================= 3) CLASIFICAR CON LOS 6 AUTOENCODERS =================
    _, _, prob_pru = pasar_por_autoencoders(limpiadores, binarios, X_pru)
    pred_ae = prob_pru.argmax(axis=1)
    print(f"\nExactitud del sistema de autoencoders: {np.mean(pred_ae == y_pru):.1%}")
    graficar_cadena(X_pru[idx], y_pru[idx], limpiadores, binarios, clases, "4_ae_cadena_limpieza_binario.png")
    graficar_matriz_probabilidades(prob_pru, y_pru, clases, "5_ae_probabilidades_prueba.png")
    graficar_matriz_confusion(y_pru, pred_ae, clases, "6 autoencoders (limpieza + binario)",
                              "6_matriz_confusion_autoencoders.png")
    res_binarios = graficar_binarios(prob_pru, y_pru, clases, "7_ae_binarios.png")

    # ================= 4) CNN =================
    print("\n[3/3] Entrenando CNN...")
    pesos = dict(enumerate(compute_class_weight("balanced", classes=np.unique(y_ent), y=y_ent)))
    cnn = construir_cnn(forma, len(clases))
    cnn.summary()
    h_cnn = cnn.fit(X_ent_a, y_ent_a, validation_data=(X_pru, y_pru), epochs=args.epocas_cnn,
                    batch_size=32, class_weight=pesos, verbose=2,
                    callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=10, restore_best_weights=True)])
    graficar_historial([("CNN", h_cnn.history)], "CNN: pérdida y exactitud", "8_cnn_entrenamiento.png")
    pred_cnn = cnn.predict(X_pru, verbose=0).argmax(1)
    graficar_matriz_confusion(y_pru, pred_cnn, clases, "CNN", "9_matriz_confusion_cnn.png")
    graficar_predicciones_prueba(X_pru, y_pru, pred_ae, pred_cnn, clases, "10_predicciones_prueba.png")

    # Recorrido de una imagen por cada filtro de la CNN
    if args.imagen:
        img = cargar_imagen(args.imagen, args.tam)
        nombre_clase = os.path.basename(os.path.dirname(args.imagen))
        clase_real = clases.index(nombre_clase) if nombre_clase in clases else -1
        ejemplos = [(img, clase_real, os.path.splitext(os.path.basename(args.imagen))[0])]
    else:  # una imagen de prueba de cada clase
        ejemplos = [(X_pru[np.where(y_pru == c)[0][0]], c, clases[c]) for c in range(len(clases))]
    for img, c, etiqueta in ejemplos:
        print(f"Visualizando recorrido por la CNN: {etiqueta}")
        activs = graficar_diagrama_cnn(cnn, img, clases, c, f"11_cnn_diagrama_{etiqueta}.png")
        graficar_mapas_por_capa({k: activs[k] for k in ["conv1", "pool1", "conv2", "pool2", "conv3", "pool3"]},
                                img, f"12_cnn_mapas_por_capa_{etiqueta}.png")
        graficar_filtros_conv1(cnn, activs["conv1"], img, f"13_cnn_filtros_conv1_{etiqueta}.png")

    # ================= 5) Imágenes nuevas (opcional) =================
    if args.nuevas:
        X_new, archivos = cargar_nuevas(args.nuevas, args.tam)
        print(f"\nImágenes nuevas: {len(archivos)}")
        if archivos:
            # Paso 1: ¿es una galaxia?
            es_galaxia, err_d = graficar_deteccion(X_new, [os.path.basename(f) for f in archivos], detector,
                                                   umbral_galaxia, "14_nuevas_detector.png")
            # Paso 2: solo las galaxias se clasifican
            _, _, prob_new = pasar_por_autoencoders(limpiadores, binarios, X_new)
            p_cnn = cnn.predict(X_new, verbose=0).argmax(1)
            for f, g, ed, pr, pc in zip(archivos, es_galaxia, err_d, prob_new, p_cnn):
                resultado = (f"6 AE -> {clases[pr.argmax()]:<11} CNN -> {clases[pc]}" if g
                             else "NO ES UNA GALAXIA (no se clasifica)")
                print(f"  {os.path.basename(f):<30} detector={ed:.3f}  {resultado}")
            if es_galaxia.any():
                graficar_cadena(X_new[es_galaxia], np.full(es_galaxia.sum(), -1), limpiadores, binarios, clases,
                                "15_nuevas_cadena_autoencoders.png")

    # ================= Resumen =================
    reporte = [f"Prueba: {n_prueba} imágenes por clase ({PORC_PRUEBA:.0%} de la clase más pequeña)",
               f"=== Detector de galaxias (umbral = {umbral_galaxia:.3f}) ===",
               f"Galaxias de prueba aceptadas: {np.sum(err_det_pru <= umbral_galaxia)} de {len(err_det_pru)}",
               f"Imágenes que NO son galaxias rechazadas: {np.sum(err_det_no > umbral_galaxia)} de {len(err_det_no)} "
               f"({', '.join(nombres_no)})", "",
               "=== Sistema de 6 autoencoders (limpieza -> binario, gana la mayor P(sí)) ===",
               classification_report(y_pru, pred_ae, labels=range(len(clases)), target_names=clases,
                                     zero_division=0),
               "--- Cada autoencoder binario por separado (clase vs. resto) ---"]
    for clase, umbral, auc, acc, cm in res_binarios:
        reporte.append(f"AE-binario {clase:<11} umbral={umbral:.5f}  AUC={auc:.2f}  exactitud={acc:.1%}  "
                       f"[VP={cm[0, 0]} FN={cm[0, 1]} FP={cm[1, 0]} VN={cm[1, 1]}]")
    reporte += ["", "=== CNN ===",
                classification_report(y_pru, pred_cnn, labels=range(len(clases)), target_names=clases,
                                      zero_division=0)]
    with open(os.path.join(CARPETA_SALIDA, "reporte.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(reporte))
    print("\n" + "\n".join(reporte))
    print(f"Listo. Revisa la carpeta '{CARPETA_SALIDA}/'.")


if __name__ == "__main__":
    main()
