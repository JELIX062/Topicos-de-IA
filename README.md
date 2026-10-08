# Clasificación de galaxias con 6 autoencoders y una CNN

`clasificador_galaxias.py` (y el cuaderno `clasificador_galaxias_colab.ipynb`) clasifica las imágenes de `data/`
(Elliptical, Espiral, Lenticular) de dos formas:

1. **Sistema de 6 autoencoders, dos por clase y en cadena:**
   - **3 autoencoders de limpieza** (de-noising): cada uno aprende, solo con su clase, a recibir una imagen con
     ruido y devolverla limpia.
   - **3 autoencoders binarios**: cada uno reconstruye las imágenes ya limpias de su clase. Si el error entre su
     salida y su entrada es menor que un umbral, la imagen "sí pertenece" a su clase.
   - Una imagen pasa por las 3 cadenas *limpieza → binario* y se asigna a la clase con **menor error**.
2. **CNN** con la misma estructura que el diagrama clásico:
   `(Convolución + ReLU → Pooling) ×3 → Flatten → Capa densa → Softmax`.
   Se dibuja cómo la imagen real pasa por cada filtro y capa hasta la probabilidad de cada clase.

**División de datos:** la prueba es el **20% de la clase más pequeña** (18 elípticas → 4 imágenes) y se toma ese
**mismo número de cada clase** (12 imágenes de prueba). El resto es entrenamiento, con rotaciones/espejos como
aumento de datos.

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
python clasificador_galaxias.py                                 # entrena y genera todas las figuras (~5 min en CPU)
python clasificador_galaxias.py --imagen data/Espiral/NGC24.jpg # visualizar una imagen en la CNN
python clasificador_galaxias.py --nuevas foto.jpg carpeta/      # clasificar imágenes nuevas al final
python clasificador_galaxias.py --help                          # épocas, ruido, tamaño del cuello de botella...
```

## Figuras generadas en `resultados/`

| Archivo | Contenido |
|---|---|
| `1_ae_limpieza_entrenamiento.png` | Curvas de los 3 autoencoders de limpieza |
| `2_ae_limpieza_antes_despues.png` | **Antes** (original y con ruido) y **después** (limpia por cada AE) |
| `3_ae_binario_entrenamiento.png` | Curvas de los 3 autoencoders binarios |
| `4_ae_cadena_limpieza_binario.png` | **Recorrido completo**: entrada → limpia → salida binaria de cada clase, errores y decisión |
| `5_ae_errores_prueba.png` | Error de cada imagen de prueba en cada autoencoder binario |
| `6_matriz_confusion_autoencoders.png` | Matriz de confusión del sistema de 6 autoencoders |
| `7_ae_binarios.png` | Cada autoencoder binario por separado: histogramas, umbral y matriz sí/no |
| `8_cnn_entrenamiento.png` | Pérdida y exactitud de la CNN |
| `9_matriz_confusion_cnn.png` | Matriz de confusión de la CNN |
| `10_predicciones_prueba.png` | Todas las imágenes de prueba con la predicción de ambos modelos |
| `11_cnn_diagrama_<clase>.png` | **Diagrama de la CNN con los mapas reales** de cada capa, flatten, capa densa y softmax |
| `12_cnn_mapas_por_capa_<clase>.png` | Los 16 mapas más activos de cada capa (conv1 → pool3) |
| `13_cnn_filtros_conv1_<clase>.png` | Cada kernel 3×3 aprendido y el mapa que produce |
| `14_nuevas_cadena_autoencoders.png` | (con `--nuevas`) recorrido de las imágenes nuevas por los 6 autoencoders |
| `reporte.txt` | Precisión, recall y F1 de ambos modelos y de cada autoencoder binario |

En Colab, la sección 7 permite elegir cualquier imagen de prueba con un formulario y la sección 8 subir imágenes
nuevas (sueltas o en `.zip`).

---

# Clasificador con un autoencoder por clase

`clasificador_autoencoders_por_clase.py` (y el cuaderno `autoencoders_por_clase_colab.ipynb`) entrena **tres
autoencoders**, cada uno solo con imágenes de su clase (AE-Elliptical, AE-Espiral, AE-Lenticular).

- **Clasificación:** una imagen nueva pasa por los tres; se compara la salida con la entrada (MSE) y gana el
  autoencoder con **menor error**, es decir, el más cercano.
- **Binario:** cada autoencoder decide "pertenece a mi clase" si su error es menor que un umbral
  (el percentil 95 del error en sus imágenes de entrenamiento).

```bash
python clasificador_autoencoders_por_clase.py                          # 20 épocas, criterio mse
python clasificador_autoencoders_por_clase.py --criterio normalizado   # divide el error entre el error típico de cada AE
python clasificador_autoencoders_por_clase.py --nuevas foto1.jpg carpeta/  # clasifica imágenes nuevas al final
```

**Imágenes nuevas:** se sigue entrenando con el 80% y evaluando con el 20%. Además, en Colab la sección 8 permite
subir imágenes nuevas (sueltas o en un `.zip`) que pasan por los tres autoencoders ya entrenados. Si se conoce su
clase se puede indicar en el formulario para ver si acertó.

Figuras en `resultados_por_clase/`: entrenamiento de los 3 AE, entrada vs. salida de cada AE, mapas de error,
errores de todo el conjunto de prueba, matriz de confusión, clasificadores binarios (histogramas + matrices
sí/no) y todas las predicciones de prueba.

> Con muchas épocas el AE-Espiral (el que tiene más imágenes y más variadas) aprende a reconstruir cualquier
> galaxia y gana siempre; por eso el valor por defecto es 20 épocas.
