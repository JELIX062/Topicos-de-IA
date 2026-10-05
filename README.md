# Clasificación de galaxias con Autoencoder y CNN

`clasificador_galaxias.py` clasifica las imágenes de `data/` (Elliptical, Espiral, Lenticular) de dos formas:

1. **Autoencoder convolucional (de-noising).** Aprende a reconstruir la galaxia a partir de una versión con ruido.
   Se muestra la imagen **antes** (original y con ruido), el código latente y la imagen **después** (reconstruida),
   junto con el error. Después se reutiliza el *encoder* para clasificar.
2. **CNN** con la misma estructura que el diagrama clásico:
   `(Convolución + ReLU → Pooling) ×3 → Flatten → Capa densa → Softmax`.
   Se dibuja cómo la imagen real pasa por cada filtro y capa hasta la probabilidad de cada clase.

## Ejecutar en Google Colab

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/jelix062/topicos-de-ia/blob/claude/autoencoders-cnn-filter-viz-bsr00n/clasificador_galaxias_colab.ipynb)

1. Abre `clasificador_galaxias_colab.ipynb` en Colab (con el botón de arriba si el repositorio es público,
   o desde Colab con `Archivo → Subir cuaderno`).
2. Opcional: `Entorno de ejecución → Cambiar tipo de entorno → GPU T4`.
3. `Entorno de ejecución → Ejecutar todo`. En el paso 1 sube `data.zip` (o elige *Google Drive* / *GitHub* en el formulario).
4. Todas las figuras se muestran debajo de cada celda; al final se descarga `resultados.zip`.

## Ejecutar en local

```bash
pip install -r requirements.txt
python clasificador_galaxias.py                                 # entrena y genera todas las figuras (~2-3 min en CPU)
python clasificador_galaxias.py --imagen data/Espiral/NGC24.jpg # visualizar una imagen en particular
python clasificador_galaxias.py --help                          # épocas, tamaño, nivel de ruido...
```

## Figuras generadas en `resultados/`

| Archivo | Contenido |
|---|---|
| `1_autoencoder_entrenamiento.png` | Curva del error de reconstrucción |
| `2_clasificador_autoencoder_entrenamiento.png` | Pérdida y exactitud del clasificador sobre el encoder |
| `3_matriz_confusion_autoencoder.png` | Matriz de confusión (autoencoder) |
| `4_autoencoder_antes_despues.png` | **Antes / después** del autoencoder + predicción |
| `5_cnn_entrenamiento.png` | Pérdida y exactitud de la CNN |
| `6_matriz_confusion_cnn.png` | Matriz de confusión (CNN) |
| `7_cnn_diagrama_<clase>.png` | **Diagrama de la CNN con los mapas reales** de cada capa, flatten, capa densa y softmax |
| `8_cnn_mapas_por_capa_<clase>.png` | Los 16 mapas más activos de cada capa (conv1 → pool3) |
| `9_cnn_filtros_conv1_<clase>.png` | Cada kernel 3×3 aprendido y el mapa que produce |
| `reporte.txt` | Precisión, recall y F1 de ambos modelos |

> Nota: el conjunto es pequeño y desbalanceado (130 espirales, 38 lenticulares, 18 elípticas), así que se usan
> rotaciones/espejos como aumento de datos y pesos por clase. Elípticas y lenticulares se parecen mucho,
> por eso son las clases que más se confunden.
