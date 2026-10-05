"""
Clasificación de galaxias (Elliptical / Espiral / Lenticular) con:

  1) Un AUTOENCODER convolucional (de-noising) que aprende a reconstruir las
     imágenes. Se muestra la imagen ANTES (original y con ruido) y DESPUÉS
     (reconstruida) del proceso. Su encoder se reutiliza para clasificar.

  2) Una CNN clásica (Convolución + ReLU -> Pooling) x3 -> Flatten -> Densa ->
     Softmax, y se grafica cómo la imagen pasa por cada filtro / capa, al
     estilo del diagrama "Convolution Neural Network (CNN)".

Uso:
    python clasificador_galaxias.py                 # entrena y genera todo
    python clasificador_galaxias.py --imagen data/Espiral/NGC24.jpg
    python clasificador_galaxias.py --epocas-ae 40 --epocas-cnn 60

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
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import keras
from keras import layers

SEMILLA = 42
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


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------
def construir_autoencoder(forma):
    entrada = keras.Input(forma, name="entrada")
    x = layers.Conv2D(32, 3, padding="same", activation="relu", name="enc_conv1")(entrada)
    x = layers.MaxPooling2D(name="enc_pool1")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="enc_conv2")(x)
    x = layers.MaxPooling2D(name="enc_pool2")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu", name="enc_conv3")(x)
    codigo = layers.MaxPooling2D(name="codigo_latente")(x)
    encoder = keras.Model(entrada, codigo, name="encoder")

    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu", name="dec_up1")(codigo)
    x = layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu", name="dec_up2")(x)
    x = layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu", name="dec_up3")(x)
    salida = layers.Conv2D(forma[-1], 3, padding="same", activation="sigmoid", name="reconstruccion")(x)
    autoencoder = keras.Model(entrada, salida, name="autoencoder")
    autoencoder.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return autoencoder, encoder


def construir_clasificador_ae(encoder, n_clases):
    """Encoder pre-entrenado (congelado) + cabeza densa con softmax."""
    encoder.trainable = False
    x = layers.Flatten(name="codigo_plano")(encoder.output)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(64, activation="relu", name="densa")(x)
    x = layers.Dropout(0.4)(x)
    salida = layers.Dense(n_clases, activation="softmax", name="softmax")(x)
    modelo = keras.Model(encoder.input, salida, name="clasificador_autoencoder")
    modelo.compile(optimizer=keras.optimizers.Adam(1e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return modelo


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


# ---------------------------------------------------------------------------
# Autoencoder: antes y después
# ---------------------------------------------------------------------------
def graficar_autoencoder_antes_despues(X, X_ruido, X_rec, codigos, probs, y, clases, nombre):
    n = len(X)
    filas = ["ANTES\nOriginal", "ANTES\nEntrada con ruido", "Código latente\n(promedio de mapas)",
             "DESPUÉS\nReconstrucción", "Error |orig - rec|"]
    fig, ejes = plt.subplots(len(filas), n, figsize=(2.3 * max(n, 2), 2.4 * len(filas)), squeeze=False)
    for j in range(n):
        imgs = [X[j, ..., 0], X_ruido[j, ..., 0], codigos[j].mean(-1),
                X_rec[j, ..., 0], np.abs(X[j, ..., 0] - X_rec[j, ..., 0])]
        cmaps = ["gray", "gray", "magma", "gray", "inferno"]
        for i, (img, cmap) in enumerate(zip(imgs, cmaps)):
            eje = ejes[i, j]
            eje.imshow(img, cmap=cmap, vmin=0 if i in (0, 1, 3) else None, vmax=1 if i in (0, 1, 3) else None)
            eje.set_xticks([])
            eje.set_yticks([])
            if j == 0:
                eje.set_ylabel(filas[i], fontsize=10)
        pred = int(np.argmax(probs[j]))
        color = "black" if y[j] < 0 else ("green" if pred == y[j] else "red")
        ejes[0, j].set_title(f"Real: {clases[y[j]] if y[j] >= 0 else 'desconocida'}", fontsize=10)
        ejes[3, j].set_title(f"Pred: {clases[pred]} ({probs[j][pred]:.0%})", fontsize=10, color=color)
    fig.suptitle("Autoencoder convolucional: imagen ANTES y DESPUÉS del proceso + clasificación", fontsize=14)
    fig.tight_layout()
    guardar(fig, nombre)


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
    p.add_argument("--epocas-ae", type=int, default=30)
    p.add_argument("--epocas-clf", type=int, default=40)
    p.add_argument("--epocas-cnn", type=int, default=40)
    p.add_argument("--ruido", type=float, default=0.15, help="ruido gaussiano para el autoencoder")
    p.add_argument("--imagen", default=None, help="imagen a visualizar en el diagrama de la CNN")
    args = p.parse_args()

    keras.utils.set_random_seed(SEMILLA)
    rng = np.random.default_rng(SEMILLA)
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    print("Cargando imágenes...")
    X, y, rutas, clases = cargar_datos(args.datos, args.tam)
    print(f"  {len(X)} imágenes, clases: " + ", ".join(f"{c}={np.sum(y == i)}" for i, c in enumerate(clases)))
    X_ent, X_pru, y_ent, y_pru, r_ent, r_pru = train_test_split(
        X, y, rutas, test_size=0.25, stratify=y, random_state=SEMILLA)
    X_ent_a, y_ent_a = aumentar(X_ent, y_ent)
    pesos = dict(enumerate(compute_class_weight("balanced", classes=np.unique(y_ent), y=y_ent)))
    forma = X.shape[1:]

    # ================= 1) AUTOENCODER =================
    print("\n[1/2] Entrenando autoencoder (de-noising)...")
    autoencoder, encoder = construir_autoencoder(forma)
    h_ae = autoencoder.fit(agregar_ruido(X_ent_a, args.ruido, rng), X_ent_a,
                           validation_data=(agregar_ruido(X_pru, args.ruido, rng), X_pru),
                           epochs=args.epocas_ae, batch_size=32, verbose=2)
    graficar_historial([("AE", h_ae.history)], "Autoencoder: error de reconstrucción (MSE)",
                       "1_autoencoder_entrenamiento.png")

    print("Entrenando clasificador sobre el código latente del autoencoder...")
    clf_ae = construir_clasificador_ae(encoder, len(clases))
    h1 = clf_ae.fit(X_ent_a, y_ent_a, validation_data=(X_pru, y_pru), epochs=args.epocas_clf,
                    batch_size=32, class_weight=pesos, verbose=2)
    # Ajuste fino: se descongela el encoder con tasa de aprendizaje baja
    encoder.trainable = True
    clf_ae.compile(optimizer=keras.optimizers.Adam(1e-4), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = clf_ae.fit(X_ent_a, y_ent_a, validation_data=(X_pru, y_pru), epochs=max(1, args.epocas_clf // 2),
                    batch_size=32, class_weight=pesos, verbose=2)
    h_clf = {k: h1.history[k] + h2.history[k] for k in h1.history}
    graficar_historial([("AE+clasif.", h_clf)], "Clasificador basado en autoencoder",
                       "2_clasificador_autoencoder_entrenamiento.png")

    pred_ae = clf_ae.predict(X_pru, verbose=0)
    graficar_matriz_confusion(y_pru, pred_ae.argmax(1), clases, "Autoencoder + clasificador",
                              "3_matriz_confusion_autoencoder.png")

    # Antes / después: 2 ejemplos por clase del conjunto de prueba
    idx = np.concatenate([np.where(y_pru == c)[0][:2] for c in range(len(clases))])
    X_ruido = agregar_ruido(X_pru[idx], args.ruido, rng)
    graficar_autoencoder_antes_despues(
        X_pru[idx], X_ruido, autoencoder.predict(X_ruido, verbose=0), encoder.predict(X_ruido, verbose=0),
        clf_ae.predict(X_pru[idx], verbose=0), y_pru[idx], clases, "4_autoencoder_antes_despues.png")

    # ================= 2) CNN =================
    print("\n[2/2] Entrenando CNN...")
    cnn = construir_cnn(forma, len(clases))
    cnn.summary()
    h_cnn = cnn.fit(X_ent_a, y_ent_a, validation_data=(X_pru, y_pru), epochs=args.epocas_cnn,
                    batch_size=32, class_weight=pesos, verbose=2,
                    callbacks=[keras.callbacks.EarlyStopping("val_loss", patience=10, restore_best_weights=True)])
    graficar_historial([("CNN", h_cnn.history)], "CNN: pérdida y exactitud", "5_cnn_entrenamiento.png")
    pred_cnn = cnn.predict(X_pru, verbose=0)
    graficar_matriz_confusion(y_pru, pred_cnn.argmax(1), clases, "CNN", "6_matriz_confusion_cnn.png")

    # Imagen a visualizar
    if args.imagen:
        img = cargar_imagen(args.imagen, args.tam)
        nombre_clase = os.path.basename(os.path.dirname(args.imagen))
        clase_real = clases.index(nombre_clase) if nombre_clase in clases else -1
        ejemplos = [(img, clase_real, os.path.splitext(os.path.basename(args.imagen))[0])]
    else:  # una imagen de prueba de cada clase
        ejemplos = []
        for c in range(len(clases)):
            i = np.where(y_pru == c)[0][0]
            ejemplos.append((X_pru[i], c, clases[c]))

    for img, c, etiqueta in ejemplos:
        print(f"Visualizando recorrido por la CNN: {etiqueta}")
        activs = graficar_diagrama_cnn(cnn, img, clases, c, f"7_cnn_diagrama_{etiqueta}.png")
        graficar_mapas_por_capa({k: activs[k] for k in ["conv1", "pool1", "conv2", "pool2", "conv3", "pool3"]},
                                img, f"8_cnn_mapas_por_capa_{etiqueta}.png")
        graficar_filtros_conv1(cnn, activs["conv1"], img, f"9_cnn_filtros_conv1_{etiqueta}.png")

    # ================= Resumen =================
    reporte = ["=== Autoencoder + clasificador ===",
               classification_report(y_pru, pred_ae.argmax(1), labels=range(len(clases)),
                                      target_names=clases, zero_division=0),
               "=== CNN ===",
               classification_report(y_pru, pred_cnn.argmax(1), labels=range(len(clases)),
                                     target_names=clases, zero_division=0)]
    with open(os.path.join(CARPETA_SALIDA, "reporte.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(reporte))
    print("\n" + "\n".join(reporte))
    print(f"Listo. Revisa la carpeta '{CARPETA_SALIDA}/'.")


if __name__ == "__main__":
    main()
