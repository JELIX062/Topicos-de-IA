# Problema 3 · Separación de frutas que se tocan o se superponen (YOLO, segmentación de instancias)

## Paso 1: dataset

`preparar_dataset_frutas.py` (o el cuaderno `dataset_frutas_colab.ipynb`) arma el dataset a partir de **COCO 2017**,
que trae máscaras por instancia de `apple`, `orange` y `banana` en fotos reales: fruteros, montones, frutas en la
mano o detrás de otros objetos.

```bash
pip install numpy pillow matplotlib
python preparar_dataset_frutas.py --solo-juntas      # ~1500 imágenes, 70/20/10
python preparar_dataset_frutas.py --help             # clases, límite de imágenes, partición, semilla...
```

Qué hace:

1. Descarga las anotaciones oficiales (~240 MB, una sola vez) y **solo las imágenes seleccionadas**, no todo COCO.
2. Filtra: al menos 2 frutas por imagen; descarta imágenes con anotaciones `iscrowd` de fruta (un montón marcado
   como un solo bloque no tiene instancias separadas); con `--solo-juntas`, exige al menos un par de frutas cuyas
   cajas se toquen o se superpongan.
3. Convierte cada fruta a una línea YOLO-seg: `clase x1 y1 x2 y2 ...` (polígono normalizado 0-1). Cuando un
   obstáculo parte una fruta en varios pedazos, COCO la guarda como varios polígonos; se unen en un solo
   contorno para que siga siendo **una** instancia.
4. Divide en train/val/test (70/20/10, semilla 42) y genera:

| Archivo en `datasets/frutas/` | Contenido |
|---|---|
| `images/{train,val,test}/`, `labels/{train,val,test}/` | Imágenes y etiquetas YOLO-seg |
| `frutas.yaml` | Configuración para Ultralytics (`data=datasets/frutas/frutas.yaml`) |
| `estadisticas.txt` / `.json` | Imágenes, instancias por clase e imágenes con frutas juntas por partición |
| `distribucion_clases.png` | Instancias por clase y partición |
| `vista_previa_instancias.png` | Ejemplos con cada instancia (máscara) en un color distinto |
| `imagen_seguimiento.png` | La imagen de prueba con más frutas juntas: máscara vs. caja de cada fruta. Sirve como la imagen que se sigue por todas las etapas |

Clases: `0: apple`, `1: orange`, `2: banana`.
