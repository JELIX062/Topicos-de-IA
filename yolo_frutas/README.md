# Problema 3 · Separación de frutas que se tocan o se superponen (YOLO, segmentación de instancias)

## Cuaderno principal: `problema3_frutas_yolo_seg_colab.ipynb`

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jelix062/topicos-de-ia/blob/claude/dazzling-lamport-ng6tti/yolo_frutas/problema3_frutas_yolo_seg_colab.ipynb)

Usa el **Fruit instance segmentation dataset for YOLOvFIS network** (Figshare, 6 clases, formato YOLOv8-seg) y
**YOLO11n-seg** ajustado con ese dataset. Sigue una misma imagen de prueba (la que tiene más frutas que se tocan)
por las 7 etapas obligatorias de la tarea:

| Sección | Qué muestra |
|---|---|
| 1. Entrada y preparación | Descarga, partición train/val/test, fotos originales vs. aumentadas (y fugas entre particiones), instancias por clase, imagen de seguimiento con sus anotaciones |
| Entrenamiento | Fine-tuning desde COCO (o carga de un `best.pt` ya entrenado) y curvas |
| 2. Preprocesamiento | Letterbox manual (redimensionar, relleno, normalización, tensor), comparado con el tensor real que usa Ultralytics, y recuperación de coordenadas |
| 3. Backbone | Hooks en todas las capas: formas reales, activaciones, kernels de la capa 0 y comprobaciones cuantitativas (bordes Sobel, activación fruta/fondo) |
| 4. Neck | Entradas y salidas reales de cada `Concat`, mapas P3/P4/P5 y diagrama con las formas reales |
| 5. Cabeza | Tensor de salida `(1, 4+nc+32, N)`, candidatos con números, los 32 prototipos, coeficientes → máscara, caja reconstruida desde la DFL |
| 6. Posprocesamiento | Umbral, xywh→xyxy, IoU calculado a mano, NMS antes/después, transformación de la máscara a la resolución original y una máscara por fruta; se verifica contra Ultralytics |
| 7. Resultado y evaluación | Resultado final, tiempos, mAP de cajas y máscaras en test, IoU por fruta, aciertos/errores, recall de frutas que se tocan vs. aisladas, experimento caja vs. máscara (áreas en píxeles) y guía para la pregunta a defender |

Las figuras dicen **[EVIDENCIA]** (tensores reales del modelo) o **[DIAGRAMA]** (explicativas) y se guardan en
`resultados_frutas/` (se descargan como zip al final).

## Alternativa anterior: dataset a partir de COCO 2017

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
