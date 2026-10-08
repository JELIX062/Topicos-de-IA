"""
YOLO (You Only Look Once) paso a paso: Backbone -> Neck -> Head.

El programa carga el modelo YOLO26n preentrenado en COCO (80 clases) y, en vez
de llamar a model.predict() como una caja negra, recorre la red CAPA POR CAPA
y guarda lo que sale de cada una para dibujarlo, siguiendo la presentación:

  1. Imagen
  2. Preprocesamiento    redimensionar (letterbox 640x640) + normalizar (0-1)
  3. Backbone            Conv + C3k2 (x4 etapas) -> SPPF -> C2PSA
                         - C3k2: división en 2 ramas (directa / bloques) + concat
                         - SPPF: Conv 1x1 -> MaxPool x3 -> concat + Conv 1x1 (+ residual)
                         - C2PSA: 50% de canales con atención, 50% directo
  4. Neck                fusión multiescala: arriba->abajo (Upsample x2 + Concat)
                         y abajo->arriba (Conv stride 2 + Concat) -> P3 / P4 / P5
  5. Detection Head      sin DFL: regresión directa de 4 valores (l, t, r, b)
                         + 80 puntuaciones de clase en cada celda de P3/P4/P5
  6. Cabeza elegida      One-to-Many + NMS   o   One-to-One (NMS-free)
  7. Predicciones        [x1, y1, x2, y2, confianza, clase]
  8. Objetos detectados

La decodificación de cajas, el NMS y la selección One-to-One están programados
aquí mismo (numpy) para que cada paso se vea; al final se comparan con el
resultado oficial de Ultralytics.

Uso:
    python yolo_visualizacion.py                       # imagen de ejemplo (bus.jpg)
    python yolo_visualizacion.py --imagen foto.jpg
    python yolo_visualizacion.py --imagen foto.jpg --conf 0.3 --iou 0.7
    python yolo_visualizacion.py --modelo yolo26s.pt   # n, s, m, l, x

Todas las figuras se guardan en la carpeta  resultados_yolo/
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

from ultralytics import YOLO
from ultralytics.utils import ASSETS

TAM = 640  # tamaño de entrada de la red
CARPETA_SALIDA = "resultados_yolo"

# Colores de la presentación
VERDE = "#00C2A8"
VERDE_OSC = "#0E9F8A"
NEGRO = "#13151A"
ROJO = "#E5484D"
GRIS = "#8A8F98"
FONDO = "#F8F8FA"

plt.rcParams.update(
    {
        "figure.facecolor": FONDO,
        "axes.facecolor": FONDO,
        "savefig.facecolor": FONDO,
        "font.size": 10,
        "axes.titlesize": 10,
    }
)

PALETA_CAJAS = ["#00C2A8", "#E5484D", "#3B82F6", "#F59E0B", "#A855F7", "#EC4899", "#10B981", "#6366F1"]


# ---------------------------------------------------------------------------
# Utilidades de dibujo
# ---------------------------------------------------------------------------
def guardar(fig, nombre):
    ruta = os.path.join(CARPETA_SALIDA, nombre)
    fig.savefig(ruta, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {ruta}")


def encabezado(fig, seccion, titulo, subtitulo, y=0.995):
    fig.text(0.01, y, seccion, color=VERDE_OSC, fontsize=10, fontweight="bold", va="top")
    fig.text(0.01, y - 0.035, titulo, color="black", fontsize=20, va="top")
    fig.text(0.01, y - 0.09, subtitulo, color="#444", fontsize=11, va="top")


def caja(ax, x, y, w, h, titulo, sub="", estilo="negro", fs=10):
    """Caja redondeada al estilo de la presentación (coordenadas de ax en 0-1)."""
    colores = {
        "negro": (NEGRO, NEGRO, "white"),
        "blanco": ("white", NEGRO, "black"),
        "verde": (VERDE, NEGRO, "black"),
        "rojo": (ROJO, NEGRO, "white"),
        "verde_borde": ("white", VERDE, "black"),
    }
    fondo, borde, texto = colores[estilo]
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012", fc=fondo, ec=borde, lw=1.6,
            transform=ax.transAxes, clip_on=False,
        )
    )
    if sub:
        ax.text(x + w / 2, y + h * 0.62, titulo, ha="center", va="center", color=texto, fontsize=fs,
                fontweight="bold", transform=ax.transAxes)
        ax.text(x + w / 2, y + h * 0.32, sub, ha="center", va="center", color=texto, fontsize=fs - 1.5,
                transform=ax.transAxes)
    else:
        ax.text(x + w / 2, y + h / 2, titulo, ha="center", va="center", color=texto, fontsize=fs,
                fontweight="bold", transform=ax.transAxes)


def flecha(ax, x1, y1, x2, y2, color=NEGRO, lw=1.8):
    ax.add_patch(
        FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=color, lw=lw,
                        transform=ax.transAxes, clip_on=False)
    )


def flecha_fig(fig, ax1, ax2, xy1=(1, 0.5), xy2=(0, 0.5), color=NEGRO):
    """Flecha entre dos ejes distintos de la figura."""
    con = ConnectionPatch(xyA=xy1, coordsA=ax1.transAxes, xyB=xy2, coordsB=ax2.transAxes, arrowstyle="-|>",
                          mutation_scale=14, color=color, lw=1.8)
    fig.add_artist(con)


def sin_ejes(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def mapa_medio(t):
    """(C,H,W) -> mapa 2D normalizado 0-1 con la activación media de todos los canales."""
    m = t.float().mean(0).cpu().numpy()
    return (m - m.min()) / (m.max() - m.min() + 1e-8)


def canales_top(t, k):
    """Índices de los k canales con mayor activación media."""
    return t.float().mean((1, 2)).argsort(descending=True)[:k].tolist()


def norm01(a):
    a = np.asarray(a, dtype="float32")
    return (a - a.min()) / (a.max() - a.min() + 1e-8)


def redimensionar(m, tam=TAM):
    return np.asarray(Image.fromarray((norm01(m) * 255).astype("uint8")).resize((tam, tam), Image.BILINEAR)) / 255.0


def redimensionar_crudo(m, tam=TAM):
    """Agranda un mapa 0-1 a tam x tam sin normalizarlo (celdas visibles)."""
    return np.asarray(Image.fromarray((np.clip(m, 0, 1) * 255).astype("uint8")).resize((tam, tam), Image.NEAREST)) / 255.0


def superponer(ax, img, mapa, titulo="", cmap="jet", alfa=0.5):
    ax.imshow(img)
    ax.imshow(redimensionar(mapa, img.shape[0]), cmap=cmap, alpha=alfa)
    ax.set_title(titulo)
    sin_ejes(ax)


def forma(t):
    return "×".join(str(s) for s in t.shape)


def dibujar_cajas(ax, cajas, nombres, color=None, lw=2.0, etiquetas=True, alfa=1.0):
    for i, (x1, y1, x2, y2, conf, cls) in enumerate(cajas):
        c = color or PALETA_CAJAS[int(cls) % len(PALETA_CAJAS)]
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=c, lw=lw, alpha=alfa))
        if etiquetas:
            ax.text(x1, y1 - 3, f"{nombres[int(cls)]} {conf:.0%}", color="black", fontsize=8, fontweight="bold",
                    va="bottom", bbox=dict(fc=c, ec="none", pad=1.5, alpha=0.95))


# ---------------------------------------------------------------------------
# 2 · Preprocesamiento
# ---------------------------------------------------------------------------
def letterbox(img, tam=TAM, relleno=114):
    """Redimensiona conservando la proporción y rellena con gris hasta tam x tam."""
    w0, h0 = img.size
    r = min(tam / w0, tam / h0)
    nw, nh = round(w0 * r), round(h0 * r)
    izq, arr = (tam - nw) // 2, (tam - nh) // 2
    lienzo = Image.new("RGB", (tam, tam), (relleno,) * 3)
    lienzo.paste(img.resize((nw, nh), Image.BILINEAR), (izq, arr))
    return np.asarray(lienzo), r, (izq, arr)


def a_tensor(img_lb):
    """HWC uint8 (0-255) -> tensor 1x3xHxW float (0-1)."""
    return torch.from_numpy(img_lb.astype("float32") / 255.0).permute(2, 0, 1)[None]


# ---------------------------------------------------------------------------
# 3-4 · Forward capa por capa (Backbone + Neck)
# ---------------------------------------------------------------------------
NOMBRES_CAPA = {
    0: "Conv s2", 1: "Conv s2", 2: "C3k2", 3: "Conv s2", 4: "C3k2", 5: "Conv s2", 6: "C3k2", 7: "Conv s2",
    8: "C3k2", 9: "SPPF", 10: "C2PSA", 11: "Upsample ×2", 12: "Concat", 13: "C3k2", 14: "Upsample ×2",
    15: "Concat", 16: "C3k2 → P3", 17: "Conv s2", 18: "Concat", 19: "C3k2 → P4", 20: "Conv s2", 21: "Concat",
    22: "C3k2 → P5",
}


@torch.no_grad()
def forward_por_capas(red, x):
    """Ejecuta la red capa por capa (igual que Ultralytics) y guarda la salida de cada una."""
    salidas = []
    for i, capa in enumerate(red.model[:-1]):  # todas menos la Detect, que se hace a mano
        if capa.f == -1:
            entrada = x
        elif isinstance(capa.f, int):
            entrada = salidas[capa.f]
        else:
            entrada = [x if j == -1 else salidas[j] for j in capa.f]
        x = capa(entrada)
        salidas.append(x)
    return salidas


@torch.no_grad()
def c3k2_interno(bloque, x):
    """Repite a mano el C3k2: Conv 1x1 -> división en 2 -> bloques -> concat -> Conv 1x1."""
    y = bloque.cv1(x)
    directa, procesada = y.chunk(2, 1)
    ramas = [directa, procesada]
    for m in bloque.m:
        ramas.append(m(ramas[-1]))
    concat = torch.cat(ramas, 1)
    salida = bloque.cv2(concat)
    return dict(entrada=x, conv1=y, directa=directa, procesada=procesada, bloques=ramas[2:], concat=concat,
                salida=salida)


@torch.no_grad()
def sppf_interno(bloque, x):
    """Repite a mano el SPPF: Conv 1x1 -> MaxPool x3 -> concat -> Conv 1x1 (+ residual)."""
    y = [bloque.cv1(x)]
    for _ in range(getattr(bloque, "n", 3)):
        y.append(bloque.m(y[-1]))
    concat = torch.cat(y, 1)
    sin_res = bloque.cv2(concat)
    usa_res = getattr(bloque, "add", False)
    salida = sin_res + x if usa_res else sin_res
    return dict(entrada=x, conv1=y[0], pools=y[1:], concat=concat, sin_residual=sin_res, salida=salida,
                residual=usa_res, k=bloque.m.kernel_size)


@torch.no_grad()
def c2psa_interno(bloque, x):
    """Repite a mano el C2PSA y extrae la matriz de atención de cada cabeza."""
    a, b = bloque.cv1(x).split((bloque.c, bloque.c), dim=1)
    atenciones = []
    z = b
    for psa in bloque.m:
        at = psa.attn
        B, C, H, W = z.shape
        N = H * W
        qkv = at.qkv(z)
        q, k, v = qkv.view(B, at.num_heads, at.key_dim * 2 + at.head_dim, N).split(
            [at.key_dim, at.key_dim, at.head_dim], dim=2)
        attn = ((q * at.scale).transpose(-2, -1) @ k).softmax(dim=-1)  # (B, cabezas, N, N)
        atenciones.append(attn[0])
        z = psa(z)
    salida = bloque.cv2(torch.cat((a, z), 1))
    return dict(entrada=x, directa=a, a_atencion=b, tras_atencion=z, atenciones=atenciones, salida=salida)


# ---------------------------------------------------------------------------
# 5-7 · Detection Head (sin DFL), NMS y One-to-One programados a mano
# ---------------------------------------------------------------------------
@torch.no_grad()
def cabeza(detect, feats, box_head, cls_head):
    """Aplica una cabeza (One-to-Many u One-to-One) a P3/P4/P5 y decodifica las cajas.

    Sin DFL (reg_max = 1): cada celda predice directamente 4 distancias (l, t, r, b)
    desde su centro hasta los bordes de la caja, en unidades de celda.
    """
    escalas = []
    cajas, puntuaciones = [], []
    for i, f in enumerate(feats):
        _, _, h, w = f.shape
        s = float(detect.stride[i])
        ltrb = box_head[i](f)[0]  # (4, h, w)
        logits = cls_head[i](f)[0]  # (80, h, w)
        prob = logits.sigmoid()
        gy, gx = torch.meshgrid(torch.arange(h) + 0.5, torch.arange(w) + 0.5, indexing="ij")
        x1 = (gx - ltrb[0]) * s
        y1 = (gy - ltrb[1]) * s
        x2 = (gx + ltrb[2]) * s
        y2 = (gy + ltrb[3]) * s
        xyxy = torch.stack([x1, y1, x2, y2], -1).view(-1, 4)
        escalas.append(dict(stride=s, h=h, w=w, ltrb=ltrb, prob=prob, xyxy=xyxy.view(h, w, 4)))
        cajas.append(xyxy)
        puntuaciones.append(prob.view(detect.nc, -1).T)
    return escalas, torch.cat(cajas).numpy(), torch.cat(puntuaciones).numpy()


def iou(caja, cajas):
    x1 = np.maximum(caja[0], cajas[:, 0])
    y1 = np.maximum(caja[1], cajas[:, 1])
    x2 = np.minimum(caja[2], cajas[:, 2])
    y2 = np.minimum(caja[3], cajas[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(caja) + area(cajas) - inter + 1e-9)


def candidatos(cajas, punt, conf):
    """Una predicción por celda: la clase con mayor puntuación, si supera conf."""
    cls = punt.argmax(1)
    p = punt.max(1)
    sel = np.where(p > conf)[0]
    sel = sel[np.argsort(-p[sel])]
    return np.column_stack([cajas[sel], p[sel], cls[sel]]), sel


def nms(dets, umbral_iou):
    """Non-Maximum Suppression por clase: si dos cajas de la misma clase se solapan
    más que umbral_iou, se queda la de mayor confianza."""
    quedan, eliminadas = [], []
    orden = list(range(len(dets)))
    while orden:
        i = orden.pop(0)
        quedan.append(i)
        if not orden:
            break
        resto = np.array(orden)
        misma = dets[resto, 5] == dets[i, 5]
        solapa = iou(dets[i, :4], dets[resto, :4]) > umbral_iou
        borrar = resto[misma & solapa]
        eliminadas += [(int(j), i) for j in borrar]
        orden = [j for j in orden if j not in set(borrar.tolist())]
    return dets[quedan], eliminadas


def one_to_one(cajas, punt, conf, max_det=300):
    """Selección NMS-free: top-k celdas por puntuación y top-k (celda, clase); sin borrar duplicados."""
    k = min(max_det, len(punt))
    top_celdas = np.argsort(-punt.max(1))[:k]
    sub = punt[top_celdas]
    plano = np.argsort(-sub.ravel())[:k]
    celda = top_celdas[plano // punt.shape[1]]
    cls = plano % punt.shape[1]
    p = sub.ravel()[plano]
    ok = p > conf
    return np.column_stack([cajas[celda[ok]], p[ok], cls[ok]]), celda[ok]


def a_original(dets, r, pad, w0, h0):
    """Pasa cajas del lienzo 640x640 a coordenadas de la imagen original."""
    d = dets.copy()
    d[:, [0, 2]] = ((d[:, [0, 2]] - pad[0]) / r).clip(0, w0)
    d[:, [1, 3]] = ((d[:, [1, 3]] - pad[1]) / r).clip(0, h0)
    return d


# ---------------------------------------------------------------------------
# Figuras
# ---------------------------------------------------------------------------
def fig_clasificar_vs_detectar(img0, finales, nombres):
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.8))
    encabezado(fig, "01 · INTRODUCCIÓN", "¿Qué problema resuelve? Clasificar vs detectar",
               "Saber qué objetos hay en una imagen y dónde están. YOLO resuelve la detección.", y=1.10)
    for ax in axs:
        ax.imshow(img0)
        sin_ejes(ax)
    if len(finales):
        mejor = finales[np.argmax(finales[:, 4])]
        clases = sorted({nombres[int(c)] for c in finales[:, 5]})
        axs[0].set_title("CLASIFICACIÓN · ¿Qué hay en la imagen?", loc="left", color=VERDE_OSC, fontweight="bold")
        axs[0].set_xlabel(f"Salida: {nombres[int(mejor[5])]} (solo la clase)", fontsize=12, color="white",
                          bbox=dict(fc=NEGRO, ec="none", pad=8))
        dibujar_cajas(axs[1], finales, nombres)
        axs[1].set_title("DETECCIÓN · ¿Qué hay y dónde está?", loc="left", color=VERDE_OSC, fontweight="bold")
        x1, y1, x2, y2, c, k = mejor
        axs[1].set_xlabel(f"Salida: {len(finales)} objetos ({', '.join(clases)}) · ej. {nombres[int(k)]} · "
                          f"caja ({x1:.0f}, {y1:.0f}, {x2 - x1:.0f}, {y2 - y1:.0f}) · {c:.0%}", fontsize=11,
                          bbox=dict(fc=VERDE, ec="none", pad=8))
    guardar(fig, "01_clasificar_vs_detectar.png")


def fig_preprocesamiento(img0, img_lb, tensor, r, pad):
    fig = plt.figure(figsize=(15, 5.6))
    encabezado(fig, "06 · INFERENCIA · PASO 2", "Preprocesamiento: redimensionar + normalizar",
               f"La red siempre recibe un tensor 1×3×{TAM}×{TAM} con valores entre 0 y 1.", y=1.08)
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.1], wspace=0.35)
    a0, a1, a2, a3 = (fig.add_subplot(gs[0, i]) for i in range(4))
    a0.imshow(img0)
    a0.set_title(f"1 · Imagen original\n{img0.shape[1]}×{img0.shape[0]} px, valores 0-255")
    a1.imshow(img_lb)
    a1.add_patch(Rectangle((pad[0], pad[1]), TAM - 2 * pad[0], TAM - 2 * pad[1], fill=False, ec=VERDE, lw=2, ls="--"))
    a1.set_title(f"2 · Letterbox {TAM}×{TAM}\nescala ×{r:.3f}, relleno gris (114)")
    canales = tensor[0].numpy()
    a2.imshow(np.hstack([canales[0], canales[1], canales[2]]), cmap="gray", vmin=0, vmax=1)
    a2.set_title("3 · Tensor 3 canales (R | G | B)\nvalores / 255 → [0, 1]")
    for ax in (a0, a1, a2):
        sin_ejes(ax)
    a3.hist(img0.ravel(), bins=50, color=GRIS, alpha=0.8, label="original (0-255)")
    a3b = a3.twiny()
    a3b.hist(canales.ravel(), bins=50, color=VERDE, alpha=0.6, label="normalizado (0-1)")
    a3.set_xlabel("valor original", color=GRIS)
    a3b.set_xlabel("valor normalizado", color=VERDE_OSC)
    a3.set_yticks([])
    a3.set_title("4 · Distribución de valores", pad=28)
    for i, ax in enumerate((a0, a1, a2)):
        flecha_fig(fig, ax, (a1, a2, a3)[i], (1.02, 0.5), (-0.04, 0.5))
    guardar(fig, "02_preprocesamiento.png")


def fig_backbone(img_lb, salidas):
    """Diagrama del backbone con los mapas reales de cada capa 0-10."""
    capas = list(range(11))
    fig = plt.figure(figsize=(20, 8.5))
    encabezado(fig, "03 · ARQUITECTURA · BACKBONE", "Backbone: de píxeles a características cada vez más abstractas",
               "Activación media de cada capa sobre la imagen. Cada Conv stride 2 reduce el mapa a la mitad y "
               "aumenta los canales.", y=1.02)
    gs = fig.add_gridspec(2, 12, height_ratios=[1, 1], hspace=0.55, wspace=0.25)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img_lb)
    ax.set_title(f"Imagen\n3×{TAM}×{TAM}", fontweight="bold")
    sin_ejes(ax)
    previo = ax
    ejes = []
    for j, i in enumerate(capas):
        t = salidas[i][0]
        ax = fig.add_subplot(gs[0, j + 1])
        ax.imshow(mapa_medio(t), cmap="viridis")
        color = VERDE_OSC if i in (4, 6, 10) else "black"
        extra = {4: "\n→ P3 (al neck)", 6: "\n→ P4 (al neck)", 10: "\n→ P5 (al neck)"}.get(i, "")
        ax.set_title(f"{i} · {NOMBRES_CAPA[i]}\n{forma(t)}{extra}", color=color, fontsize=9)
        sin_ejes(ax)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_color(VERDE if i in (4, 6, 10) else NEGRO)
            s.set_linewidth(2 if i in (4, 6, 10) else 0.8)
        flecha_fig(fig, previo, ax, (1.02, 0.5), (-0.02, 0.5))
        previo = ax
        ejes.append(ax)
    # Fila inferior: superposición sobre la imagen de las etapas clave
    claves = [(2, "Etapa 1 (C3k2)\nbordes y texturas"), (4, "Etapa 2 (C3k2)\npartes pequeñas"),
              (6, "Etapa 3 (C3k2)\npartes de objetos"), (8, "Etapa 4 (C3k2)\nobjetos completos"),
              (9, "SPPF\ncontexto multiescala"), (10, "C2PSA\natención espacial")]
    for j, (i, txt) in enumerate(claves):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        superponer(ax, img_lb, mapa_medio(salidas[i][0]), f"{i} · {txt}\n{forma(salidas[i][0])}")
    guardar(fig, "03_backbone.png")


def fig_backbone_canales(salidas, k=8):
    capas = list(range(11))
    fig, axs = plt.subplots(len(capas), k, figsize=(k * 1.6, len(capas) * 1.75))
    fig.suptitle(f"Backbone: los {k} canales (filtros) más activos de cada capa", fontsize=15, y=1.0)
    for f, i in enumerate(capas):
        t = salidas[i][0]
        for c, ch in enumerate(canales_top(t, k)):
            ax = axs[f, c]
            ax.imshow(t[ch].numpy(), cmap="viridis")
            sin_ejes(ax)
            if c == 0:
                ax.set_ylabel(f"{i} · {NOMBRES_CAPA[i]}\n{forma(t)}", rotation=0, ha="right", va="center",
                              fontsize=8.5)
            ax.set_title(f"canal {ch}", fontsize=7, color=GRIS)
    fig.tight_layout()
    guardar(fig, "04_backbone_canales_por_capa.png")


def fig_c3k2(img_lb, d, idx):
    fig = plt.figure(figsize=(19, 7.2))
    encabezado(fig, "03 · ARQUITECTURA · C3k2", f"C3k2 (capa {idx}): extraer más, gastar menos",
               "Divide las características: solo una parte pasa por los bloques convolucionales; la otra va "
               "directa y al final se reúnen.", y=1.03)
    gs = fig.add_gridspec(2, 6, wspace=0.45, hspace=0.6)
    a_in = fig.add_subplot(gs[:, 0])
    a_c1 = fig.add_subplot(gs[:, 1])
    a_dir = fig.add_subplot(gs[0, 2:4])
    a_blq = fig.add_subplot(gs[1, 2:4])
    a_cat = fig.add_subplot(gs[:, 4])
    a_out = fig.add_subplot(gs[:, 5])
    superponer(a_in, img_lb, mapa_medio(d["entrada"][0]), f"Entrada\n{forma(d['entrada'][0])}")
    a_c1.imshow(mapa_medio(d["conv1"][0]), cmap="viridis")
    a_c1.set_title(f"Conv 1×1\n{forma(d['conv1'][0])}\ndivisión en 2", fontweight="bold")
    n = len(d["bloques"])
    a_dir.imshow(mapa_medio(d["directa"][0]), cmap="viridis")
    a_dir.set_title(f"Rama directa (pasa sin procesar) · {forma(d['directa'][0])}", fontweight="bold")
    mapas = [mapa_medio(d["procesada"][0])] + [mapa_medio(b[0]) for b in d["bloques"]]
    sep = np.ones((mapas[0].shape[0], 2))
    a_blq.imshow(np.hstack(sum([[m, sep] for m in mapas], [])[:-1]), cmap="viridis")
    pasos_txt = "bloque 1" if n == 1 else f"bloque 1 … bloque {n}"
    a_blq.set_title(f"Bloques convolucionales: entrada → {pasos_txt} (C3k / Bottleneck ×{n})", fontweight="bold")
    a_cat.imshow(mapa_medio(d["concat"][0]), cmap="viridis")
    a_cat.set_title(f"Combinación\nconcat {forma(d['concat'][0])}\n+ Conv 1×1", fontweight="bold")
    superponer(a_out, img_lb, mapa_medio(d["salida"][0]), f"Salida\n{forma(d['salida'][0])}")
    for s in a_out.spines.values():
        s.set_visible(True)
        s.set_color(VERDE)
        s.set_linewidth(3)
    for ax in (a_c1, a_dir, a_blq, a_cat):
        sin_ejes(ax)
    flecha_fig(fig, a_in, a_c1, (1.03, 0.5), (-0.03, 0.5))
    flecha_fig(fig, a_c1, a_dir, (1.03, 0.6), (-0.02, 0.5))
    flecha_fig(fig, a_c1, a_blq, (1.03, 0.4), (-0.02, 0.5))
    flecha_fig(fig, a_dir, a_cat, (1.02, 0.5), (-0.03, 0.6))
    flecha_fig(fig, a_blq, a_cat, (1.02, 0.5), (-0.03, 0.4))
    flecha_fig(fig, a_cat, a_out, (1.03, 0.5), (-0.03, 0.5))
    c_total = d["conv1"].shape[1]
    fig.text(0.5, -0.02, f"Idea clave: de los {c_total} canales, solo {c_total // 2} pasan por los bloques; "
             "reunir ambas ramas conserva información con menos cómputo.", ha="center", color=GRIS, fontsize=12)
    guardar(fig, f"05_c3k2_interno_capa{idx}.png")


def fig_sppf(img_lb, d):
    fig = plt.figure(figsize=(20, 6.6))
    encabezado(fig, "03 · ARQUITECTURA · SPPF", "SPPF + conexión residual",
               "Añade contexto de la imagen a varias escalas espaciales: cada MaxPool cubre una región mayor.", y=1.05)
    pasos = [("Feature Map\n(entrada)", d["entrada"]), ("Conv 1×1", d["conv1"])]
    k = d["k"]
    for j, p in enumerate(d["pools"]):
        campo = 1 + (j + 1) * (k - 1)
        pasos.append((f"MaxPool {j + 1}\n(ventana efectiva {campo}×{campo})", p))
    pasos.append(("Concat\n+ Conv 1×1", d["sin_residual"]))
    pasos.append(("Feature Map\nenriquecido (+ entrada)" if d["residual"] else "Feature Map\nenriquecido",
                  d["salida"]))
    gs = fig.add_gridspec(2, len(pasos), height_ratios=[1, 1], hspace=0.45, wspace=0.35, top=0.72)
    previo = None
    for j, (titulo, t) in enumerate(pasos):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(mapa_medio(t[0]), cmap="viridis")
        final = j == len(pasos) - 1
        ax.set_title(f"{titulo}\n{forma(t[0])}", fontweight="bold", color=VERDE_OSC if final else "black")
        sin_ejes(ax)
        if final:
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color(VERDE)
                s.set_linewidth(3)
        if previo is not None:
            flecha_fig(fig, previo, ax, (1.03, 0.5), (-0.03, 0.5))
        previo = ax
        ax2 = fig.add_subplot(gs[1, j])
        superponer(ax2, img_lb, mapa_medio(t[0]), "")
    if d["residual"]:
        primero = fig.axes[0]
        fig.add_artist(ConnectionPatch(xyA=(0.5, 1.4), coordsA=primero.transAxes, xyB=(0.5, 1.4),
                                       coordsB=previo.transAxes, arrowstyle="-", color=VERDE, lw=1.8))
        for ax_ in (primero, previo):
            fig.add_artist(ConnectionPatch(xyA=(0.5, 1.4), coordsA=ax_.transAxes, xyB=(0.5, 1.33),
                                           coordsB=ax_.transAxes, arrowstyle="-|>" if ax_ is previo else "-",
                                           color=VERDE, lw=1.8))
        y_txt = primero.get_position().y1 + 0.42 * primero.get_position().height
        fig.text(0.5, y_txt, "conexión residual (entrada + salida)", ha="center", color=VERDE_OSC)
    fig.text(0.02, -0.03, "• MaxPool en cadena: cada pasada cubre una región mayor (contexto local, medio y "
             "amplio).    • Residual: se suma la entrada original para no perder detalle fino.", color=GRIS,
             fontsize=11)
    guardar(fig, "06_sppf.png")


def fig_c2psa(img_lb, d, finales_lb, nombres):
    at = d["atenciones"][0]  # (cabezas, N, N)
    cabezas, N, _ = at.shape
    h, w = d["entrada"].shape[2:]
    fig = plt.figure(figsize=(20, 11))
    encabezado(fig, "03 · ARQUITECTURA · C2PSA", "C2PSA: atención sensible a la posición",
               "PSA = Position-Sensitive Attention: la red pondera cada región según su relevancia. La mitad de los "
               "canales pasa por atención y la otra mitad directa.", y=1.0)
    gs = fig.add_gridspec(3, 6, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1, 1, 1], wspace=0.4, hspace=0.55,
                          top=0.86)
    # Fila superior: lo que ocurre dentro de la atención
    recibida = at.mean(0).mean(0).view(h, w).numpy()
    a_rec = fig.add_subplot(gs[0, 2])
    superponer(a_rec, img_lb, recibida, f"Dentro de la atención:\natención recibida ({cabezas} cabezas, {N} pos.)",
               cmap="inferno", alfa=0.6)
    if len(finales_lb):
        x1, y1, x2, y2, c, k = finales_lb[np.argmax(finales_lb[:, 4])]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        etiqueta = nombres[int(k)]
    else:
        cx = cy = TAM / 2
        etiqueta = "centro"
    gx = int(np.clip(cx / TAM * w, 0, w - 1))
    gy = int(np.clip(cy / TAM * h, 0, h - 1))
    fila = at.mean(0)[gy * w + gx].view(h, w).numpy()
    a_q = fig.add_subplot(gs[0, 3])
    superponer(a_q, img_lb, fila, f"¿A dónde mira la celda\ndel «{etiqueta}»? (★)", cmap="inferno", alfa=0.6)
    a_q.plot(cx, cy, marker="*", color="white", markersize=16, markeredgecolor="black")
    # Filas 2-3: el bloque C2PSA
    a_in = fig.add_subplot(gs[1:, 0])
    superponer(a_in, img_lb, mapa_medio(d["entrada"][0]), f"Features\n{forma(d['entrada'][0])}")
    a_at = fig.add_subplot(gs[1, 1])
    a_at.imshow(mapa_medio(d["a_atencion"][0]), cmap="viridis")
    a_at.set_title(f"50% canales → atención\n{forma(d['a_atencion'][0])}", fontweight="bold")
    a_tr = fig.add_subplot(gs[1, 2:4])
    a_tr.imshow(np.hstack([mapa_medio(d["a_atencion"][0]), np.ones((h, 1)), mapa_medio(d["tras_atencion"][0])]),
                cmap="viridis")
    a_tr.set_title("Atención (PSA + FFN): antes  |  después", fontweight="bold")
    a_dir = fig.add_subplot(gs[2, 1])
    a_dir.imshow(mapa_medio(d["directa"][0]), cmap="viridis")
    a_dir.set_title(f"50% canales → directo\nsin procesar · {forma(d['directa'][0])}", fontweight="bold")
    a_cat = fig.add_subplot(gs[1:, 4])
    a_cat.imshow(mapa_medio(torch.cat([d["directa"], d["tras_atencion"]], 1)[0]), cmap="viridis")
    a_cat.set_title("Concat (C2)\n+ Conv 1×1", fontweight="bold", color=VERDE_OSC)
    a_out = fig.add_subplot(gs[1:, 5])
    superponer(a_out, img_lb, mapa_medio(d["salida"][0]), f"Salida\n{forma(d['salida'][0])}")
    for ax in (a_at, a_dir, a_tr, a_cat):
        sin_ejes(ax)
    flecha_fig(fig, a_in, a_at, (1.03, 0.6), (-0.03, 0.5))
    flecha_fig(fig, a_in, a_dir, (1.03, 0.4), (-0.03, 0.5))
    flecha_fig(fig, a_at, a_tr, (1.03, 0.5), (-0.02, 0.5))
    flecha_fig(fig, a_tr, a_cat, (1.02, 0.5), (-0.03, 0.6))
    flecha_fig(fig, a_dir, a_cat, (1.03, 0.5), (-0.03, 0.4))
    flecha_fig(fig, a_cat, a_out, (1.03, 0.5), (-0.03, 0.5))
    flecha_fig(fig, a_tr, a_rec, (0.25, 1.25), (0.5, -0.03), color=GRIS)
    fig.text(0.5, 0.04, "La mitad de los canales pasa por atención y la otra mitad pasa directa: más foco en los "
             "objetos con poco costo extra.", ha="center", color=GRIS, fontsize=12)
    guardar(fig, "07_c2psa_atencion.png")


def fig_neck(img_lb, salidas):
    """Neck como en la diapositiva: filas P5/P4/P3, arriba→abajo y abajo→arriba con mapas reales."""
    fig = plt.figure(figsize=(17, 12))
    encabezado(fig, "03 · ARQUITECTURA · NECK", "Neck: fusión multiescala",
               "Dos caminos que mezclan significado (semántica) y detalle (resolución) para detectar objetos de "
               "todos los tamaños.", y=1.07)
    gs = fig.add_gridspec(3, 4, wspace=0.55, hspace=0.45, top=0.84)
    filas = {
        "P5": dict(back=10, td=10, bu=22, td_txt="Semántica\npunto de partida", bu_txt="Conv stride 2 + Concat",
                   out="P5 · objetos grandes"),
        "P4": dict(back=6, td=13, bu=19, td_txt="Upsample ×2 + Concat\n(+ C3k2)", bu_txt="Conv stride 2 + Concat\n(+ C3k2)",
                   out="P4 · objetos medianos"),
        "P3": dict(back=4, td=16, bu=16, td_txt="Upsample ×2 + Concat\n(+ C3k2)", bu_txt="Detalle espacial\npunto de partida",
                   out="P3 · objetos pequeños"),
    }
    ejes = {}
    for f, (nombre, c) in enumerate(filas.items()):
        tam_txt = {"P5": "mapa pequeño", "P4": "mapa mediano", "P3": "mapa grande"}[nombre]
        a_b = fig.add_subplot(gs[f, 0])
        a_b.imshow(mapa_medio(salidas[c["back"]][0]), cmap="viridis")
        a_b.set_title(f"Backbone · {tam_txt}\ncapa {c['back']} · {forma(salidas[c['back']][0])}", fontweight="bold")
        a_t = fig.add_subplot(gs[f, 1])
        a_t.imshow(mapa_medio(salidas[c["td"]][0]), cmap="viridis")
        a_t.set_title(f"{c['td_txt']}\ncapa {c['td']} · {forma(salidas[c['td']][0])}", fontweight="bold")
        a_u = fig.add_subplot(gs[f, 2])
        a_u.imshow(mapa_medio(salidas[c["bu"]][0]), cmap="viridis")
        a_u.set_title(f"{c['bu_txt']}\ncapa {c['bu']} · {forma(salidas[c['bu']][0])}", fontweight="bold")
        a_o = fig.add_subplot(gs[f, 3])
        superponer(a_o, img_lb, mapa_medio(salidas[c["bu"]][0]), c["out"])
        a_o.title.set_fontweight("bold")
        a_o.title.set_color(VERDE_OSC)
        for ax in (a_b, a_t, a_u):
            sin_ejes(ax)
        for s in a_o.spines.values():
            s.set_visible(True)
            s.set_color(VERDE)
            s.set_linewidth(3)
        flecha_fig(fig, a_b, a_t, (1.04, 0.5), (-0.04, 0.5))
        flecha_fig(fig, a_t, a_u, (1.04, 0.5), (-0.04, 0.5))
        flecha_fig(fig, a_u, a_o, (1.04, 0.5), (-0.04, 0.5))
        ejes[nombre] = (a_t, a_u)
    flecha_fig(fig, ejes["P5"][0], ejes["P4"][0], (0.5, -0.03), (0.5, 1.32))
    flecha_fig(fig, ejes["P4"][0], ejes["P3"][0], (0.5, -0.03), (0.5, 1.32))
    flecha_fig(fig, ejes["P3"][1], ejes["P4"][1], (0.5, 1.32), (0.5, -0.03))
    flecha_fig(fig, ejes["P4"][1], ejes["P5"][1], (0.5, 1.32), (0.5, -0.03))
    pos_t = ejes["P5"][0].get_position()
    pos_u = ejes["P5"][1].get_position()
    fig.text((pos_t.x0 + pos_t.x1) / 2, 0.895, "ARRIBA → ABAJO: lleva semántica", ha="center", color=VERDE_OSC,
             fontweight="bold")
    fig.text((pos_u.x0 + pos_u.x1) / 2, 0.895, "ABAJO → ARRIBA: lleva detalle", ha="center", color=VERDE_OSC,
             fontweight="bold")
    fig.text(0.5, 0.06, "Upsample: agranda el mapa al doble.  Concat: une dos mapas de características en uno.  "
             "Conv stride 2: reduce el mapa a la mitad.", ha="center", color=GRIS, fontsize=12)
    guardar(fig, "08_neck_fusion_multiescala.png")


def fig_head_mapas(img_lb, esc_m, esc_o, nombres):
    """Puntuación de clase de cada celda en P3, P4 y P5 para las dos cabezas."""
    fig, axs = plt.subplots(2, 4, figsize=(19, 10))
    encabezado(fig, "04 · DETECTION HEAD", "Detection Head: cada celda de P3 / P4 / P5 predice clase y caja",
               f"Mapa = máxima probabilidad de las {len(nombres)} clases en cada celda (sigmoide). "
               f"{sum(e['h'] * e['w'] for e in esc_m)} celdas en total.", y=1.04)
    for f, (escalas, nombre) in enumerate([(esc_m, "One-to-Many"), (esc_o, "One-to-One")]):
        for j, e in enumerate(escalas):
            ax = axs[f, j]
            m = e["prob"].max(0)[0].numpy()
            ax.imshow(img_lb)
            ax.imshow(np.asarray(Image.fromarray((m * 255).astype("uint8")).resize((TAM, TAM), Image.NEAREST)) / 255,
                      cmap="inferno", alpha=0.6, vmin=0, vmax=1)
            p = ["P3", "P4", "P5"][j]
            obj = ["pequeños", "medianos", "grandes"][j]
            ax.set_title(f"{nombre} · {p} ({e['h']}×{e['w']} celdas, stride {e['stride']:.0f})\nobjetos {obj} · "
                         f"máx {m.max():.2f}", fontweight="bold", color=VERDE_OSC if f else "black")
            sin_ejes(ax)
        # clase dominante por celda en la escala más útil
        ax = axs[f, 3]
        m_all = [e["prob"].max(0)[0].numpy() for e in escalas]
        mejor = int(np.argmax([m.max() for m in m_all]))
        e = escalas[mejor]
        val, cls = e["prob"].max(0)
        cls = cls.numpy()
        mascara = val.numpy() > 0.25
        ax.imshow(img_lb)
        usadas = sorted(set(cls[mascara].tolist()))
        cmap = plt.get_cmap("tab10")
        rgba = np.zeros((*cls.shape, 4))
        for n, c in enumerate(usadas):
            rgba[(cls == c) & mascara] = (*cmap(n % 10)[:3], 0.6)
        ax.imshow(np.asarray(Image.fromarray((rgba * 255).astype("uint8")).resize((TAM, TAM), Image.NEAREST)))
        for n, c in enumerate(usadas[:8]):
            ax.plot([], [], "s", color=cmap(n % 10), label=nombres[c])
        if usadas:
            ax.legend(loc="lower right", fontsize=8)
        ax.set_title(f"{nombre} · clase dominante en {['P3', 'P4', 'P5'][mejor]}\n(celdas con prob > 0.25)",
                     fontweight="bold")
        sin_ejes(ax)
    guardar(fig, "09_head_mapas_de_clase.png")


def fig_head_regresion(img_lb, escalas, nombres, max_ejemplos=3):
    """Regresión directa (sin DFL): 4 valores l, t, r, b desde el centro de la celda."""
    ejemplos = []
    for j, e in enumerate(escalas):
        val, cls = e["prob"].max(0)
        idx = int(val.flatten().argmax())
        gy, gx = divmod(idx, e["w"])
        ejemplos.append((float(val.flatten()[idx]), j, gy, gx, int(cls.flatten()[idx])))
    ejemplos.sort(reverse=True)
    ejemplos = ejemplos[:max_ejemplos]
    fig, axs = plt.subplots(1, len(ejemplos) + 1, figsize=(6 * (len(ejemplos) + 1), 6.6),
                            gridspec_kw=dict(width_ratios=[1] * len(ejemplos) + [0.9]))
    encabezado(fig, "04 · DETECTION HEAD · SIN DFL", "Regresión directa: 4 valores por celda",
               "YOLO26 elimina DFL: cada celda predice directamente las distancias izquierda, arriba, derecha y "
               "abajo desde su centro hasta los bordes del objeto.", y=1.1)
    for ax, (p, j, gy, gx, c) in zip(axs, ejemplos):
        e = escalas[j]
        s = e["stride"]
        l, t, r, b = e["ltrb"][:, gy, gx].tolist()
        cx, cy = (gx + 0.5) * s, (gy + 0.5) * s
        ax.imshow(img_lb)
        # cuadrícula de la escala
        for k in range(0, e["w"] + 1):
            ax.axvline(k * s, color="white", lw=0.3, alpha=0.4)
        for k in range(0, e["h"] + 1):
            ax.axhline(k * s, color="white", lw=0.3, alpha=0.4)
        ax.add_patch(Rectangle((gx * s, gy * s), s, s, fc=VERDE, ec="black", alpha=0.7))
        x1, y1, x2, y2 = cx - l * s, cy - t * s, cx + r * s, cy + b * s
        ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=VERDE, lw=2.5))
        kw = dict(arrowstyle="-|>", mutation_scale=14, lw=2)
        for (dx, dy), col, txt in [((x1, cy), ROJO, f"l={l:.2f}"), ((x2, cy), ROJO, f"r={r:.2f}"),
                                   ((cx, y1), "#3B82F6", f"t={t:.2f}"), ((cx, y2), "#3B82F6", f"b={b:.2f}")]:
            ax.add_patch(FancyArrowPatch((cx, cy), (dx, dy), color=col, **kw))
            ax.text((cx + dx) / 2, (cy + dy) / 2, txt, color="white", fontsize=9, fontweight="bold",
                    bbox=dict(fc=col, ec="none", pad=1.5))
        ax.plot(cx, cy, "o", color="white", markeredgecolor="black")
        ax.set_title(f"{['P3', 'P4', 'P5'][j]} · celda ({gx}, {gy}) · stride {s:.0f}\n{nombres[c]} {p:.0%}",
                     fontweight="bold")
        ax.set_xlim(0, TAM)
        ax.set_ylim(TAM, 0)
        sin_ejes(ax)
    ax = axs[-1]
    ax.axis("off")
    p, j, gy, gx, c = ejemplos[0]
    e = escalas[j]
    s = e["stride"]
    l, t, r, b = e["ltrb"][:, gy, gx].tolist()
    cx, cy = (gx + 0.5) * s, (gy + 0.5) * s
    texto = (
        "Cálculo de la caja (ejemplo 1)\n\n"
        f"centro de la celda  = (({gx}+0.5)·{s:.0f}, ({gy}+0.5)·{s:.0f})\n"
        f"                    = ({cx:.0f}, {cy:.0f}) px\n\n"
        f"salida de la cabeza = [l, t, r, b]\n"
        f"                    = [{l:.2f}, {t:.2f}, {r:.2f}, {b:.2f}]\n\n"
        f"x1 = {cx:.0f} − {l:.2f}·{s:.0f} = {cx - l * s:.0f}\n"
        f"y1 = {cy:.0f} − {t:.2f}·{s:.0f} = {cy - t * s:.0f}\n"
        f"x2 = {cx:.0f} + {r:.2f}·{s:.0f} = {cx + r * s:.0f}\n"
        f"y2 = {cy:.0f} + {b:.2f}·{s:.0f} = {cy + b * s:.0f}\n\n"
        "YOLO11 (con DFL): 4 bordes × 16 bins\n"
        "  → distribución → valor esperado\n"
        "YOLO26 (sin DFL): 4 valores directos"
    )
    ax.text(0, 0.95, texto, family="monospace", fontsize=10.5, va="top",
            bbox=dict(fc="white", ec=VERDE, lw=2, boxstyle="round,pad=0.8"))
    guardar(fig, "10_head_regresion_directa.png")


def fig_nms_vs_one2one(img_lb, cand_m, final_m, eliminadas, cand_o, final_o, nombres, conf, umbral_iou):
    fig, axs = plt.subplots(2, 3, figsize=(19, 13))
    encabezado(fig, "04 · DETECTION HEAD", "Dual-head: One-to-Many + NMS  vs  One-to-One (NMS-free)",
               "NMS borra cajas duplicadas: si dos cajas de la misma clase se solapan mucho (IoU alto), queda la de "
               "mayor confianza.", y=1.0)
    for ax in axs.ravel():
        ax.imshow(img_lb)
        sin_ejes(ax)
    # Fila 1: One-to-Many
    dibujar_cajas(axs[0, 0], cand_m, nombres, etiquetas=False, lw=1.0, alfa=0.7)
    axs[0, 0].set_title(f"One-to-Many · Predicciones\n{len(cand_m)} cajas con confianza > {conf} "
                        "(varias por objeto)", fontweight="bold")
    borradas = cand_m[[i for i, _ in eliminadas]] if eliminadas else np.zeros((0, 6))
    dibujar_cajas(axs[0, 1], borradas, nombres, color=ROJO, etiquetas=False, lw=1.0, alfa=0.8)
    dibujar_cajas(axs[0, 1], final_m, nombres, color=VERDE, etiquetas=False, lw=2.5)
    axs[0, 1].set_title(f"NMS (IoU > {umbral_iou}) · filtra duplicados\nrojo: {len(borradas)} eliminadas · "
                        f"verde: {len(final_m)} se quedan", fontweight="bold", color=ROJO)
    dibujar_cajas(axs[0, 2], final_m, nombres)
    axs[0, 2].set_title(f"Resultado final (con NMS)\n{len(final_m)} objetos", fontweight="bold")
    # Fila 2: One-to-One
    dibujar_cajas(axs[1, 0], cand_o, nombres, etiquetas=False, lw=1.0, alfa=0.9)
    axs[1, 0].set_title(f"One-to-One · Predicciones\n{len(cand_o)} cajas con confianza > {conf} "
                        "(una por objeto)", fontweight="bold", color=VERDE_OSC)
    axs[1, 1].axis("off")
    axs[1, 1].imshow(np.full_like(img_lb, 248))
    caja(axs[1, 1], 0.1, 0.42, 0.8, 0.16, "NMS-free · salida directa", "sin post-procesado", estilo="verde_borde",
         fs=13)
    flecha(axs[1, 1], 0.0, 0.5, 0.1, 0.5, color=VERDE)
    flecha(axs[1, 1], 0.9, 0.5, 1.0, 0.5, color=VERDE)
    axs[1, 1].text(0.5, 0.25, "• Sin post-procesado\n• Latencia más predecible\n• Exportación más sencilla",
                   transform=axs[1, 1].transAxes, ha="center", va="top", fontsize=12, color=GRIS)
    dibujar_cajas(axs[1, 2], final_o, nombres)
    axs[1, 2].set_title(f"Resultado final (NMS-free)\n{len(final_o)} objetos", fontweight="bold", color=VERDE_OSC)
    guardar(fig, "11_nms_vs_nms_free.png")


def fig_resultado(img0, finales, nombres, titulo_cabeza):
    fig = plt.figure(figsize=(16, 8))
    encabezado(fig, "06 · INFERENCIA · PASOS 7-8", "Predicciones → Objetos detectados",
               f"Salida: [x1, y1, x2, y2, confianza, clase] en coordenadas de la imagen original ({titulo_cabeza}).",
               y=1.05)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1])
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img0)
    dibujar_cajas(ax, finales, nombres)
    sin_ejes(ax)
    at = fig.add_subplot(gs[0, 1])
    at.axis("off")
    filas = [[f"{x1:.0f}", f"{y1:.0f}", f"{x2:.0f}", f"{y2:.0f}", f"{c:.2f}", f"{int(k)} · {nombres[int(k)]}"]
             for x1, y1, x2, y2, c, k in finales[:20]]
    if filas:
        tabla = at.table(cellText=filas, colLabels=["x1", "y1", "x2", "y2", "confianza", "clase"], loc="upper center",
                         cellLoc="center")
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(10)
        tabla.scale(1, 1.5)
        for (r, c), celda in tabla.get_celld().items():
            if r == 0:
                celda.set_facecolor(NEGRO)
                celda.set_text_props(color="white", fontweight="bold")
            else:
                celda.set_facecolor("white")
                celda.set_edgecolor("#bbb")
    else:
        at.text(0.5, 0.5, "No se detectó ningún objeto", ha="center")
    guardar(fig, "12_resultado_final.png")


def fig_pipeline(img0, img_lb, salidas, esc_o, final_o_lb, finales, nombres):
    """La diapositiva '¿Qué sucede cuando YOLO26 recibe una imagen?' con miniaturas reales."""
    fig = plt.figure(figsize=(22, 11.5))
    encabezado(fig, "06 · INFERENCIA COMPLETA", "¿Qué sucede cuando YOLO26 recibe una imagen?",
               "Síntesis de todo lo visto: de píxeles a objetos detectados (miniaturas reales de esta imagen).",
               y=1.0)
    gs = fig.add_gridspec(4, 4, height_ratios=[0.33, 1, 0.33, 1], hspace=0.25, wspace=0.3)
    pasos = [
        ("1 · Imagen", f"{img0.shape[1]}×{img0.shape[0]} px", "blanco"),
        ("2 · Preprocesamiento", "redimensionar + normalizar", "blanco"),
        ("3 · Backbone", "extracción de características", "negro"),
        ("4 · Neck", "fusión multiescala", "negro"),
        ("5 · Detection Head", "sin DFL", "negro"),
        ("6 · Cabeza elegida", "One-to-One (nms=False)", "verde"),
        ("7 · Predicciones", "caja + clase + confianza", "negro"),
        ("8 · Objetos detectados", f"{len(finales)} objetos", "verde"),
    ]
    ejes_img = []
    for n, (t, s, est) in enumerate(pasos):
        f, c = (0 if n < 4 else 2), n % 4
        axt = fig.add_subplot(gs[f, c])
        axt.axis("off")
        caja(axt, 0.02, 0.05, 0.96, 0.9, t, s, estilo=est, fs=12)
        ax = fig.add_subplot(gs[f + 1, c])
        ejes_img.append(ax)
        sin_ejes(ax)
    a = ejes_img
    a[0].imshow(img0)
    a[1].imshow(img_lb)
    a[1].set_xlabel(f"1×3×{TAM}×{TAM}, valores 0-1")
    # backbone: mosaico de las 4 etapas
    mos = [redimensionar(mapa_medio(salidas[i][0]), 160) for i in (2, 4, 6, 10)]
    a[2].imshow(np.vstack([np.hstack(mos[:2]), np.hstack(mos[2:])]), cmap="viridis")
    a[2].set_xlabel("capas 2 · 4 · 6 · 10 (160→20 px de mapa)")
    mos = [redimensionar(mapa_medio(salidas[i][0]), 210) for i in (16, 19, 22)]
    a[3].imshow(np.hstack(mos), cmap="viridis")
    a[3].set_xlabel("P3 80×80 · P4 40×40 · P5 20×20")
    prob = np.maximum.reduce([redimensionar_crudo(e["prob"].max(0)[0].numpy()) for e in esc_o])
    a[4].imshow(img_lb)
    a[4].imshow(prob, cmap="inferno", alpha=0.65, vmin=0, vmax=1)
    a[4].set_xlabel("probabilidad de clase por celda (máx. de P3 · P4 · P5)")
    a[5].imshow(img_lb)
    dibujar_cajas(a[5], final_o_lb, nombres, etiquetas=False, lw=1.5)
    a[5].set_xlabel("una caja por objeto, sin NMS")
    a[6].axis("off")
    lineas = ["[x1, y1, x2, y2, conf, clase]"] + [
        f"[{x1:4.0f},{y1:4.0f},{x2:4.0f},{y2:4.0f}, {c:.2f}, {nombres[int(k)][:10]}]" for x1, y1, x2, y2, c, k in
        finales[:9]]
    if len(finales) > 9:
        lineas.append(f"... ({len(finales)} en total)")
    a[6].text(0.02, 0.98, "\n".join(lineas), family="monospace", fontsize=9, va="top", transform=a[6].transAxes)
    a[7].imshow(img0)
    dibujar_cajas(a[7], finales, nombres)
    guardar(fig, "00_pipeline_completo.png")


# ---------------------------------------------------------------------------
# Programa principal
# ---------------------------------------------------------------------------
def main():
    global CARPETA_SALIDA
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imagen", default=str(ASSETS / "bus.jpg"), help="imagen a analizar (por defecto bus.jpg)")
    p.add_argument("--modelo", default="yolo26n.pt", help="pesos YOLO26: yolo26n.pt, yolo26s.pt, ...")
    p.add_argument("--conf", type=float, default=0.25, help="umbral de confianza")
    p.add_argument("--iou", type=float, default=0.7, help="umbral IoU del NMS (cabeza One-to-Many)")
    p.add_argument("--salida", default=CARPETA_SALIDA, help="carpeta donde guardar las figuras")
    args = p.parse_args()
    CARPETA_SALIDA = args.salida
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    torch.set_grad_enabled(False)

    print(f"[1] Imagen: {args.imagen}")
    img = Image.open(args.imagen).convert("RGB")
    img0 = np.asarray(img)
    h0, w0 = img0.shape[:2]

    print(f"[ ] Cargando {args.modelo} (se descarga la primera vez)")
    modelo = YOLO(args.modelo)
    red = modelo.model.float().eval()
    detect = red.model[-1]
    nombres = red.names
    n_param = sum(p.numel() for p in red.parameters())
    if getattr(detect, "one2one_cv2", None) is None:
        raise SystemExit("Este programa necesita un modelo end-to-end con dos cabezas (YOLO26 / YOLOv10).")

    print("[2] Preprocesamiento: letterbox + normalización")
    img_lb, r, pad = letterbox(img)
    x = a_tensor(img_lb)

    print("[3-4] Backbone y Neck capa por capa")
    salidas = forward_por_capas(red, x)
    d_c3k2 = {i: c3k2_interno(red.model[i], salidas[i - 1]) for i in (2, 8)}
    d_sppf = sppf_interno(red.model[9], salidas[8])
    d_c2psa = c2psa_interno(red.model[10], salidas[9])
    for i, d in d_c3k2.items():
        assert torch.allclose(d["salida"], salidas[i], atol=1e-4)
    assert torch.allclose(d_sppf["salida"], salidas[9], atol=1e-4)
    assert torch.allclose(d_c2psa["salida"], salidas[10], atol=1e-4)

    print("[5] Detection Head (sin DFL) en P3 / P4 / P5")
    feats = [salidas[j] for j in detect.f]
    esc_m, cajas_m, punt_m = cabeza(detect, feats, detect.cv2, detect.cv3)
    esc_o, cajas_o, punt_o = cabeza(detect, feats, detect.one2one_cv2, detect.one2one_cv3)

    print("[6] One-to-Many + NMS  y  One-to-One (NMS-free)")
    cand_m, _ = candidatos(cajas_m, punt_m, args.conf)
    final_m, eliminadas = nms(cand_m, args.iou)
    cand_o, _ = candidatos(cajas_o, punt_o, args.conf)
    final_o, _ = one_to_one(cajas_o, punt_o, args.conf, detect.max_det)

    print("[7-8] Predicciones en coordenadas de la imagen original")
    finales = a_original(final_o, r, pad, w0, h0)
    finales_m = a_original(final_m, r, pad, w0, h0)

    # Comprobación contra Ultralytics (cabeza One-to-One, la que usa por defecto)
    oficial = modelo.predict(img0[..., ::-1].copy(), imgsz=TAM, conf=args.conf, verbose=False)[0].boxes
    oficial = np.column_stack([oficial.xyxy.numpy(), oficial.conf.numpy(), oficial.cls.numpy()])

    print("Generando figuras:")
    fig_pipeline(img0, img_lb, salidas, esc_o, final_o, finales, nombres)
    fig_clasificar_vs_detectar(img0, finales, nombres)
    fig_preprocesamiento(img0, img_lb, x, r, pad)
    fig_backbone(img_lb, salidas)
    fig_backbone_canales(salidas)
    for i, d in d_c3k2.items():
        fig_c3k2(img_lb, d, i)
    fig_sppf(img_lb, d_sppf)
    fig_c2psa(img_lb, d_c2psa, final_o, nombres)
    fig_neck(img_lb, salidas)
    fig_head_mapas(img_lb, esc_m, esc_o, nombres)
    fig_head_regresion(img_lb, esc_o, nombres)
    fig_nms_vs_one2one(img_lb, cand_m, final_m, eliminadas, cand_o, final_o, nombres, args.conf, args.iou)
    fig_resultado(img0, finales, nombres, "cabeza One-to-One, NMS-free")

    # Reporte de texto
    lineas = [
        f"Modelo: {args.modelo}  ·  {n_param / 1e6:.2f} M parámetros  ·  {len(nombres)} clases (COCO)",
        f"Imagen: {args.imagen}  ({w0}x{h0})  ->  letterbox {TAM}x{TAM}, escala {r:.4f}, relleno {pad}",
        "",
        "Capa  Bloque              Entrada desde   Salida",
    ]
    for i, capa in enumerate(red.model[:-1]):
        lineas.append(f"{i:>4}  {NOMBRES_CAPA[i]:<18}  {str(capa.f):<14}  {forma(salidas[i][0])}")
    lineas.append(f"{len(red.model) - 1:>4}  Detect (dual-head)  {str(detect.f):<14}  "
                  f"{cajas_o.shape[0]} celdas x (4 caja + {detect.nc} clases)")
    lineas += [
        "",
        f"Cabeza One-to-Many: {len(cand_m)} candidatos > {args.conf} -> NMS (IoU {args.iou}) elimina "
        f"{len(eliminadas)} -> {len(final_m)} objetos",
        f"Cabeza One-to-One : {len(cand_o)} candidatos > {args.conf} -> sin NMS -> {len(final_o)} objetos",
        "",
        "Detecciones finales (One-to-One, NMS-free)  [x1, y1, x2, y2, confianza, clase]:",
    ]
    lineas += [f"  [{x1:7.1f}, {y1:7.1f}, {x2:7.1f}, {y2:7.1f}, {c:.3f}, {nombres[int(k)]}]"
               for x1, y1, x2, y2, c, k in finales]
    lineas += ["", "Detecciones One-to-Many + NMS:"]
    lineas += [f"  [{x1:7.1f}, {y1:7.1f}, {x2:7.1f}, {y2:7.1f}, {c:.3f}, {nombres[int(k)]}]"
               for x1, y1, x2, y2, c, k in finales_m]
    lineas += ["", f"Comprobación: model.predict() oficial de Ultralytics -> {len(oficial)} objetos"]
    lineas += [f"  [{x1:7.1f}, {y1:7.1f}, {x2:7.1f}, {y2:7.1f}, {c:.3f}, {nombres[int(k)]}]"
               for x1, y1, x2, y2, c, k in oficial]
    lineas.append("  (pueden variar ligeramente: Ultralytics usa un letterbox rectangular, aquí uno cuadrado 640x640)")
    reporte = "\n".join(lineas)
    with open(os.path.join(CARPETA_SALIDA, "reporte.txt"), "w", encoding="utf-8") as f:
        f.write(reporte + "\n")
    print()
    print(reporte)


if __name__ == "__main__":
    main()
