"""
Problema 3 (Tarea YOLO): separación de frutas que se tocan o se superponen.

Arma un dataset de SEGMENTACIÓN DE INSTANCIAS en formato YOLO (Ultralytics) a partir de COCO 2017:

1. Descarga las anotaciones oficiales de COCO (instances_train2017.json / instances_val2017.json).
2. Se queda con las imágenes que tienen frutas (por defecto apple, orange y banana) y que cumplen:
   - al menos MIN_FRUTAS frutas anotadas una por una (cada fruta con su propio polígono),
   - ninguna anotación "iscrowd" de fruta (un montón marcado como un solo bloque arruinaría las instancias),
   - (opcional, --solo-juntas) al menos un par de frutas que se tocan o se superponen.
3. Descarga solo esas imágenes (no todo COCO).
4. Convierte cada polígono a una línea YOLO-seg:  clase x1 y1 x2 y2 ... (coordenadas normalizadas 0-1).
   Una fruta tapada por un obstáculo (hoja, mano, otra fruta) viene en COCO como VARIOS polígonos;
   se unen en uno solo para que siga siendo UNA instancia.
5. Divide en train / val / test, escribe frutas.yaml, estadísticas y figuras de vista previa.

Uso:
    python preparar_dataset_frutas.py                         # apple, orange, banana; hasta 1500 imágenes
    python preparar_dataset_frutas.py --solo-juntas           # solo imágenes con frutas que se tocan
    python preparar_dataset_frutas.py --clases apple orange --max-imagenes 800
"""

import argparse
import json
import random
import shutil
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

URL_ANOTACIONES = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
URL_IMAGEN = "http://images.cocodataset.org/{split}/{nombre}"
NOMBRES_ES = {"apple": "manzana", "orange": "naranja", "banana": "platano"}


# ------------------------------------------------------------------ descarga
def descargar(url, destino):
    destino = Path(destino)
    if destino.exists():
        return destino
    tmp = destino.with_suffix(destino.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        leido, siguiente = 0, 0
        while bloque := r.read(1 << 20):
            f.write(bloque)
            leido += len(bloque)
            if total and leido / total >= siguiente:
                print(f"\r  {destino.name}: {leido / 1e6:.0f}/{total / 1e6:.0f} MB", end="", flush=True)
                siguiente += 0.05
    print()
    tmp.rename(destino)
    return destino


def cargar_anotaciones(carpeta_cache):
    carpeta_cache.mkdir(parents=True, exist_ok=True)
    jsons = {s: carpeta_cache / f"instances_{s}.json" for s in ("train2017", "val2017")}
    if not all(p.exists() for p in jsons.values()):
        print("Descargando anotaciones de COCO 2017 (~240 MB, solo la primera vez)...")
        zip_path = descargar(URL_ANOTACIONES, carpeta_cache / "annotations_trainval2017.zip")
        with zipfile.ZipFile(zip_path) as z:
            for s, p in jsons.items():
                with z.open(f"annotations/instances_{s}.json") as src, open(p, "wb") as dst:
                    shutil.copyfileobj(src, dst)
        zip_path.unlink()
    datos = {}
    for s, p in jsons.items():
        print(f"Leyendo {p.name}...")
        with open(p) as f:
            datos[s] = json.load(f)
    return datos


# ------------------------------------------------------------------ geometría
def se_tocan(b1, b2, margen=0.02):
    """Cajas COCO [x, y, w, h]. Se consideran juntas si se intersectan (con un margen de 2% del tamaño)."""
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    mx, my = margen * max(w1, w2), margen * max(h1, h2)
    return x1 - mx < x2 + w2 and x2 - mx < x1 + w1 and y1 - my < y2 + h2 and y2 - my < y1 + h1


def iou_cajas(b1, b2):
    x1, y1, w1, h1 = b1
    x2, y2, w2, h2 = b2
    iw = max(0.0, min(x1 + w1, x2 + w2) - max(x1, x2))
    ih = max(0.0, min(y1 + h1, y2 + h2) - max(y1, y2))
    inter = iw * ih
    return inter / (w1 * h1 + w2 * h2 - inter + 1e-9)


def _indice_minimo(a, b):
    d = ((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)
    return np.unravel_index(np.argmin(d), d.shape)


def unir_segmentos(segmentos):
    """Une los pedazos de una fruta tapada en un solo contorno, conectando los puntos más cercanos
    entre pedazos consecutivos (mismo método que merge_multi_segment de Ultralytics)."""
    segmentos = [np.array(s, dtype=float).reshape(-1, 2) for s in segmentos]
    indices = [[] for _ in segmentos]
    for i in range(1, len(segmentos)):
        i1, i2 = _indice_minimo(segmentos[i - 1], segmentos[i])
        indices[i - 1].append(i1)
        indices[i].append(i2)
    s = []
    for k in range(2):
        if k == 0:  # ida
            for i, idx in enumerate(indices):
                if len(idx) == 2 and idx[0] > idx[1]:
                    idx = idx[::-1]
                    segmentos[i] = segmentos[i][::-1, :]
                segmentos[i] = np.roll(segmentos[i], -idx[0], axis=0)
                segmentos[i] = np.concatenate([segmentos[i], segmentos[i][:1]])
                if i in (0, len(indices) - 1):
                    s.append(segmentos[i])
                else:
                    idx = [0, idx[1] - idx[0]]
                    s.append(segmentos[i][idx[0]: idx[1] + 1])
        else:  # regreso
            for i in range(len(indices) - 1, -1, -1):
                if i not in (0, len(indices) - 1):
                    idx = indices[i]
                    s.append(segmentos[i][abs(idx[1] - idx[0]):])
    return np.concatenate(s)


def poligono_yolo(segmentacion, ancho, alto):
    """Lista de polígonos COCO (en píxeles) -> arreglo (N, 2) normalizado 0-1, o None si no sirve."""
    partes = [p for p in segmentacion if len(p) >= 6]
    if not partes:
        return None
    pts = unir_segmentos(partes) if len(partes) > 1 else np.array(partes[0], dtype=float).reshape(-1, 2)
    pts = pts / np.array([ancho, alto])
    return np.clip(pts, 0, 1)


# ------------------------------------------------------------------ selección
def seleccionar(datos, clases, min_frutas, solo_juntas):
    candidatas = []
    for split, d in datos.items():
        id_a_nombre = {c["id"]: c["name"] for c in d["categories"]}
        cat_ids = {c["id"] for c in d["categories"] if c["name"] in clases}
        faltan = set(clases) - {id_a_nombre[c] for c in cat_ids}
        if faltan:
            raise SystemExit(f"Estas clases no existen en COCO: {sorted(faltan)}")
        imagenes = {im["id"]: im for im in d["images"]}
        por_imagen = {}
        for a in d["annotations"]:
            if a["category_id"] in cat_ids:
                por_imagen.setdefault(a["image_id"], []).append(a)
        for img_id, anns in por_imagen.items():
            if any(a["iscrowd"] for a in anns):  # montón anotado como bloque: no hay instancias separadas
                continue
            anns = [a for a in anns if isinstance(a["segmentation"], list) and a["segmentation"]]
            if len(anns) < min_frutas:
                continue
            pares = [(i, j) for i in range(len(anns)) for j in range(i + 1, len(anns))
                     if se_tocan(anns[i]["bbox"], anns[j]["bbox"])]
            if solo_juntas and not pares:
                continue
            im = imagenes[img_id]
            candidatas.append({
                "split_coco": split, "archivo": im["file_name"], "ancho": im["width"], "alto": im["height"],
                "anotaciones": [{"clase": id_a_nombre[a["category_id"]], "bbox": a["bbox"],
                                 "segmentacion": a["segmentation"]} for a in anns],
                "pares_juntos": len(pares),
                "max_iou": max((iou_cajas(anns[i]["bbox"], anns[j]["bbox"]) for i, j in pares), default=0.0),
            })
    return candidatas


def dividir(candidatas, fracciones, semilla):
    random.Random(semilla).shuffle(candidatas)
    n = len(candidatas)
    n_train = round(fracciones[0] * n)
    n_val = round(fracciones[1] * n)
    return {"train": candidatas[:n_train], "val": candidatas[n_train:n_train + n_val],
            "test": candidatas[n_train + n_val:]}


# ------------------------------------------------------------------ escritura
def escribir_dataset(particiones, clases, salida, hilos):
    for parte, items in particiones.items():
        (salida / "images" / parte).mkdir(parents=True, exist_ok=True)
        (salida / "labels" / parte).mkdir(parents=True, exist_ok=True)

    def procesar(args):
        parte, it = args
        destino = salida / "images" / parte / it["archivo"]
        try:
            descargar(URL_IMAGEN.format(split=it["split_coco"], nombre=it["archivo"]), destino)
        except Exception as e:  # imagen que no bajó: se omite del dataset
            print(f"  aviso: no se pudo descargar {it['archivo']} ({e})")
            return False
        lineas = []
        for a in it["anotaciones"]:
            pts = poligono_yolo(a["segmentacion"], it["ancho"], it["alto"])
            if pts is not None and len(pts) >= 3:
                lineas.append(f"{clases.index(a['clase'])} " + " ".join(f"{v:.6f}" for v in pts.reshape(-1)))
        (salida / "labels" / parte / it["archivo"]).with_suffix(".txt").write_text("\n".join(lineas) + "\n")
        return True

    trabajos = [(p, it) for p, items in particiones.items() for it in items]
    print(f"Descargando {len(trabajos)} imágenes y escribiendo etiquetas YOLO-seg...")
    with ThreadPoolExecutor(hilos) as ex:
        ok = []
        for k, r in enumerate(ex.map(procesar, trabajos), 1):
            ok.append(r)
            if k % 100 == 0 or k == len(trabajos):
                print(f"  {k}/{len(trabajos)}")
    fallidas = {id(it) for (p, it), r in zip(trabajos, ok) if not r}
    return {p: [it for it in items if id(it) not in fallidas] for p, items in particiones.items()}


def escribir_yaml(salida, clases):
    lineas = [f"path: {salida.resolve()}", "train: images/train", "val: images/val", "test: images/test",
              "", "names:"] + [f"  {i}: {c}" for i, c in enumerate(clases)]
    (salida / "frutas.yaml").write_text("\n".join(lineas) + "\n")


def estadisticas(particiones, clases, salida):
    info = {"clases": clases, "particiones": {}}
    texto = ["Dataset de frutas (segmentación de instancias, formato YOLO-seg) a partir de COCO 2017", ""]
    texto.append(f"{'partición':<10}{'imágenes':>10}{'instancias':>12}{'con frutas juntas':>20}  "
                 + "  ".join(f"{c:>8}" for c in clases))
    for parte, items in particiones.items():
        conteo = Counter(a["clase"] for it in items for a in it["anotaciones"])
        juntas = sum(1 for it in items if it["pares_juntos"] > 0)
        n_inst = sum(conteo.values())
        info["particiones"][parte] = {"imagenes": len(items), "instancias": n_inst,
                                      "imagenes_con_frutas_juntas": juntas,
                                      "instancias_por_clase": {c: conteo[c] for c in clases}}
        texto.append(f"{parte:<10}{len(items):>10}{n_inst:>12}{juntas:>20}  "
                     + "  ".join(f"{conteo[c]:>8}" for c in clases))
    # Imagen para seguir durante TODO el recorrido (la tarea lo exige): la de prueba con más frutas juntas.
    test = sorted(particiones["test"], key=lambda it: (it["pares_juntos"], it["max_iou"]), reverse=True)
    if test:
        it = test[0]
        info["imagen_seguimiento"] = f"images/test/{it['archivo']}"
        texto += ["", f"Imagen de seguimiento sugerida: images/test/{it['archivo']} "
                      f"({len(it['anotaciones'])} frutas, {it['pares_juntos']} pares que se tocan, "
                      f"IoU máx. entre cajas = {it['max_iou']:.2f})"]
    (salida / "estadisticas.json").write_text(json.dumps(info, indent=2, ensure_ascii=False))
    (salida / "estadisticas.txt").write_text("\n".join(texto) + "\n")
    print("\n" + "\n".join(texto))
    return info


def figuras(salida, particiones, clases, info):
    """Vista previa: cada fruta con un color distinto (instancias) y la imagen de seguimiento por separado."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon, Rectangle
    from PIL import Image

    def dibujar(ax, parte, archivo, numerar=False):
        img = Image.open(salida / "images" / parte / archivo).convert("RGB")
        w, h = img.size
        ax.imshow(img)
        etiqueta = (salida / "labels" / parte / archivo).with_suffix(".txt")
        filas = [l.split() for l in etiqueta.read_text().splitlines() if l.strip()]
        colores = plt.cm.tab20(np.linspace(0, 1, max(len(filas), 1)))
        for k, f in enumerate(filas):
            pts = np.array(f[1:], dtype=float).reshape(-1, 2) * [w, h]
            ax.add_patch(Polygon(pts, closed=True, fc=(*colores[k][:3], 0.45), ec=colores[k], lw=1.5))
            if numerar:
                x0, y0 = pts.min(0)
                x1, y1 = pts.max(0)
                ax.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=colores[k], ls="--", lw=1))
                ax.text(x0, y0, f"{k}:{NOMBRES_ES.get(clases[int(f[0])], clases[int(f[0])])}",
                        color="white", fontsize=8, bbox=dict(fc=colores[k], ec="none", pad=1))
        ax.set_title(f"{archivo}  ({len(filas)} frutas)", fontsize=8)
        ax.axis("off")
        return len(filas)

    muestras = sorted(particiones["train"], key=lambda it: it["pares_juntos"], reverse=True)[:40]
    random.Random(0).shuffle(muestras)
    fig, axs = plt.subplots(3, 4, figsize=(16, 11))
    for ax, it in zip(axs.flat, muestras):
        dibujar(ax, "train", it["archivo"])
    for ax in axs.flat[len(muestras):]:
        ax.axis("off")
    fig.suptitle("Ejemplos de entrenamiento: cada color es una instancia (máscara) distinta", fontsize=13)
    fig.tight_layout()
    fig.savefig(salida / "vista_previa_instancias.png", dpi=110)
    plt.close(fig)

    if "imagen_seguimiento" in info:
        fig, ax = plt.subplots(figsize=(9, 7))
        dibujar(ax, "test", Path(info["imagen_seguimiento"]).name, numerar=True)
        fig.suptitle("Imagen de seguimiento: máscara (relleno) vs. caja (punteada) de cada fruta", fontsize=12)
        fig.tight_layout()
        fig.savefig(salida / "imagen_seguimiento.png", dpi=120)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    partes = list(particiones)
    x = np.arange(len(clases))
    for k, parte in enumerate(partes):
        v = [info["particiones"][parte]["instancias_por_clase"][c] for c in clases]
        ax.bar(x + (k - 1) * 0.27, v, 0.27, label=f"{parte} ({info['particiones'][parte]['imagenes']} imágenes)")
    ax.set_xticks(x, [NOMBRES_ES.get(c, c) for c in clases])
    ax.set_ylabel("instancias (frutas anotadas)")
    ax.set_title("Instancias por clase y partición")
    ax.legend()
    fig.tight_layout()
    fig.savefig(salida / "distribucion_clases.png", dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--clases", nargs="+", default=["apple", "orange", "banana"],
                   help="Clases de COCO a usar (en inglés). Otras frutas/verduras: broccoli, carrot")
    p.add_argument("--min-frutas", type=int, default=2, help="Mínimo de frutas por imagen")
    p.add_argument("--solo-juntas", action="store_true",
                   help="Solo imágenes con al menos un par de frutas que se tocan o se superponen")
    p.add_argument("--max-imagenes", type=int, default=1500, help="Límite total de imágenes (0 = todas)")
    p.add_argument("--particion", nargs=3, type=float, default=[0.7, 0.2, 0.1], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--salida", type=Path, default=Path("datasets/frutas"))
    p.add_argument("--cache", type=Path, default=Path("datasets/coco_anotaciones"))
    p.add_argument("--hilos", type=int, default=16)
    p.add_argument("--semilla", type=int, default=42)
    args = p.parse_args()

    if abs(sum(args.particion) - 1) > 1e-6:
        raise SystemExit("--particion debe sumar 1")
    datos = cargar_anotaciones(args.cache)
    candidatas = seleccionar(datos, args.clases, args.min_frutas, args.solo_juntas)
    del datos
    print(f"Imágenes que cumplen los filtros: {len(candidatas)}")
    if args.max_imagenes and len(candidatas) > args.max_imagenes:
        # Se priorizan las imágenes con frutas juntas, que son el objetivo del problema.
        random.Random(args.semilla).shuffle(candidatas)
        candidatas.sort(key=lambda it: it["pares_juntos"] > 0, reverse=True)
        candidatas = candidatas[:args.max_imagenes]
        print(f"Se usan {len(candidatas)} (prioridad a las que tienen frutas juntas)")

    particiones = dividir(candidatas, args.particion, args.semilla)
    particiones = escribir_dataset(particiones, args.clases, args.salida, args.hilos)
    escribir_yaml(args.salida, args.clases)
    info = estadisticas(particiones, args.clases, args.salida)
    figuras(args.salida, particiones, args.clases, info)
    print(f"\nListo. Dataset en {args.salida.resolve()}  ->  entrena con data='{args.salida / 'frutas.yaml'}'")


if __name__ == "__main__":
    main()
