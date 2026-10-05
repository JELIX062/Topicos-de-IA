# Texto para pedirle a una IA que me ayude con la presentación y a estudiar

> Copia todo lo que está debajo de la línea y pégalo en la IA. Si puedes, adjunta también
> `clasificador_galaxias.py` (o el cuaderno `.ipynb`) y algunas de las imágenes de `resultados/`.

---

Hola. Soy estudiante de la materia **Tópicos de Inteligencia Artificial** y tengo un programa en Python que
clasifica imágenes de galaxias usando un **autoencoder** y una **red neuronal convolucional (CNN)**. Necesito
tu ayuda con dos cosas:

1. **Organizar una presentación** (unos 12 a 15 minutos) que explique las funciones del programa y las gráficas que genera.
2. **Estudiar y entender el programa** a fondo para poder explicarlo y responder preguntas del profesor.

Abajo te describo con detalle qué hace el código y qué muestra cada gráfica.

## 1. Objetivo del programa

Clasificar imágenes astronómicas de galaxias en tres clases: **Elliptical (elíptica)**, **Espiral** y
**Lenticular**. Se comparan dos enfoques:

- **Autoencoder convolucional + clasificador:** primero una red aprende a comprimir y reconstruir las imágenes
  (sin usar las etiquetas). Después se reutiliza la parte que comprime (*encoder*) para clasificar.
- **CNN clásica:** una red convolucional entrenada directamente para clasificar, con la estructura típica
  `(Convolución + ReLU → Pooling) ×3 → Flatten → Capa densa → Softmax`.

Además de clasificar, el programa **visualiza el proceso**: cómo se ve la imagen antes y después del autoencoder,
y cómo la imagen se transforma al pasar por cada filtro y cada capa de la CNN.

Está hecho con **TensorFlow/Keras**, **NumPy**, **Matplotlib**, **Pillow** y **scikit-learn**. Hay una versión
como script (`clasificador_galaxias.py`) y otra como cuaderno de **Google Colab**.

## 2. Los datos

- 186 imágenes en escala de grises, en carpetas por clase: **130 espirales, 38 lenticulares y 18 elípticas**.
  Es un conjunto **pequeño y desbalanceado**.
- **Preprocesamiento** (`cargar_imagen`, `cargar_datos`): cada imagen se convierte a escala de grises, se recorta
  el centro para hacerla cuadrada, se reduce a **64×64 píxeles** y los valores se normalizan de 0–255 a **0–1**.
- **División 80/20** (`train_test_split` estratificado): 148 imágenes para entrenar y 38 para prueba. La división
  es **estratificada**, es decir, cada clase conserva su proporción en los dos grupos. Las 38 de prueba no se usan
  para ajustar los pesos.
- **Aumento de datos** (`aumentar`): como una galaxia no tiene "arriba" ni "abajo", cada imagen de
  entrenamiento se rota 0°, 90°, 180° y 270° y también se refleja (espejo). Así salen 8 versiones por imagen
  y las 148 se convierten en **1,184 imágenes de entrenamiento**.
- **Pesos por clase** (`compute_class_weight("balanced")`): equivocarse con una elíptica "cuesta" más que con una
  espiral, para que el modelo no aprenda a decir siempre "Espiral" (la clase mayoritaria).

## 3. Modelo 1: Autoencoder convolucional (de-noising)

**Qué es:** un autoencoder es una red que aprende a **copiar su entrada a su salida** pasando por un "cuello de
botella". Al obligarla a comprimir, aprende las características más importantes de la imagen. Es **aprendizaje no
supervisado**, porque no usa etiquetas.

**Variante de-noising (quitar ruido):** a la entrada se le agrega **ruido gaussiano** (`agregar_ruido`, factor 0.15)
y la red debe reconstruir la imagen **original limpia**. Esto la obliga a aprender la forma real de la galaxia y no
a memorizar píxeles.

**Arquitectura (`construir_autoencoder`):**
- **Encoder (comprime):** Conv2D 32 filtros 3×3 + ReLU → MaxPooling → Conv2D 64 → MaxPooling → Conv2D 64 →
  MaxPooling. La imagen pasa de **64×64×1** a un **código latente de 8×8×64**.
- **Decoder (reconstruye):** tres capas `Conv2DTranspose` con stride 2 (64, 64, 32 filtros) que vuelven a
  agrandar la imagen hasta 64×64, y una Conv2D final con activación **sigmoide** para que la salida quede entre 0 y 1.
- **Pérdida:** error cuadrático medio (**MSE**) entre la imagen original y la reconstruida. Optimizador **Adam** (tasa 0.001). 30 épocas.

**Clasificador basado en el autoencoder (`construir_clasificador_ae`):**
- Se toma el **encoder ya entrenado** y encima se pone: Flatten → Dropout 0.5 → Dense 64 (ReLU) → Dropout 0.4 → Dense 3 (**Softmax**).
- **Entrenamiento en dos fases (transfer learning):**
  1. Encoder **congelado** (sus pesos no cambian): solo se entrena la parte nueva, 40 épocas.
  2. **Ajuste fino (fine-tuning):** se descongela el encoder y se entrena todo con una tasa de aprendizaje
     más baja (0.0001), 20 épocas.
- Pérdida: **sparse categorical crossentropy**. Métrica: **accuracy (exactitud)**.

## 4. Modelo 2: CNN (red neuronal convolucional)

**Arquitectura (`construir_cnn`)**, igual al diagrama clásico de CNN:

| Capa | Qué hace | Tamaño de salida |
|---|---|---|
| Entrada | Imagen en grises | 64×64×1 |
| conv1 + ReLU | 16 filtros de 3×3 detectan bordes y brillo | 64×64×16 |
| pool1 | MaxPooling 2×2: se queda con el valor máximo de cada bloque y reduce a la mitad | 32×32×16 |
| conv2 + ReLU | 32 filtros: combinan bordes en formas (núcleo, brazos) | 32×32×32 |
| pool2 | MaxPooling 2×2 | 16×16×32 |
| conv3 + ReLU | 64 filtros: patrones más abstractos | 16×16×64 |
| pool3 | MaxPooling 2×2 | 8×8×64 |
| Flatten | "Aplana" los mapas en un vector | 4,096 valores |
| Dropout 0.5 | Apaga neuronas al azar para evitar sobreajuste | |
| Densa + ReLU | Capa totalmente conectada | 64 neuronas |
| Dropout 0.3 | | |
| Softmax | Convierte las salidas en probabilidades que suman 1 | 3 clases |

- Optimizador Adam (tasa 0.0005), hasta 40 épocas con **EarlyStopping**: se detiene si la pérdida de
  validación no mejora en 10 épocas y se queda con los mejores pesos.
- La primera parte (convoluciones y pooling) es la **extracción de características**. La parte densa es la
  **clasificación** y la softmax da la **distribución de probabilidad**.

## 5. Las gráficas que genera (carpeta `resultados/`)

1. **`1_autoencoder_entrenamiento.png`: curva de aprendizaje del autoencoder.** Error de reconstrucción (MSE) por
   época, en entrenamiento (línea continua) y validación (punteada). Si las dos bajan y se mantienen cerca,
   el autoencoder está aprendiendo sin sobreajustarse.

2. **`2_clasificador_autoencoder_entrenamiento.png`: entrenamiento del clasificador del autoencoder.** Pérdida y
   exactitud por época. Se nota el cambio entre la fase con el encoder congelado y la de ajuste fino.

3. **`3_matriz_confusion_autoencoder.png`: matriz de confusión del autoencoder.** Filas = clase real,
   columnas = clase predicha. La diagonal son los aciertos y fuera de la diagonal están las confusiones
   (por ejemplo, elípticas clasificadas como lenticulares).

4. **`4_autoencoder_antes_despues.png`: ANTES y DESPUÉS del autoencoder** (2 galaxias por clase). Filas:
   - *Antes: original*, la imagen real.
   - *Antes: entrada con ruido*, lo que recibe la red.
   - *Código latente*, el promedio de los 64 mapas de 8×8: la imagen comprimida.
   - *Después: reconstrucción*, lo que la red devuelve, ya sin ruido. Encima aparece la clase predicha y su
     probabilidad, en verde si acierta y en rojo si falla.
   - *Error |original − reconstrucción|*: dónde se equivoca más. Normalmente en estrellas pequeñas y en detalles
     finos como los brazos espirales.

5. **`5_cnn_entrenamiento.png`: curvas de pérdida y exactitud de la CNN**, en entrenamiento y validación.

6. **`6_matriz_confusion_cnn.png`: matriz de confusión de la CNN.**

7. **`7_cnn_diagrama_<clase>.png`: diagrama de la CNN con datos reales** (la gráfica principal). Reproduce el
   dibujo clásico de una CNN, pero cada "hoja" es el **mapa de características real** que produce un filtro
   cuando se le da esa galaxia:
   - **Entrada** con un cuadro amarillo que representa el **kernel** (filtro 3×3) recorriendo la imagen.
   - **Pilas de mapas** de conv1, pool1, conv2, pool2, conv3 y pool3. Se ve cómo los mapas se hacen
     **más pequeños** (64 → 32 → 16 → 8) pero **más numerosos** (16 → 32 → 64 filtros).
   - **Flatten:** una columna con los valores más altos del vector de 4,096.
   - **Capa totalmente conectada:** 12 de las 64 neuronas, con el color según su activación.
   - **Salida Softmax:** la probabilidad de cada clase y la clase predicha resaltada.
   - Abajo, las secciones *Extracción de características*, *Clasificación* y *Distribución probabilística*.

8. **`8_cnn_mapas_por_capa_<clase>.png`: mapas de características por capa.** Una fila por capa (conv1 a pool3)
   con los 16 filtros más activos. Se ve la progresión: en las primeras capas los mapas se parecen a la galaxia
   (brillo, bordes). En las últimas son manchas abstractas que indican *dónde* hay un patrón importante.

9. **`9_cnn_filtros_conv1_<clase>.png`: kernels de la primera capa.** Los 16 filtros 3×3 que aprendió conv1
   (rojo = peso positivo, azul = negativo) y debajo de cada uno el mapa que produce. Algunos filtros
   resaltan el núcleo brillante, otros los bordes y otros quedan casi apagados por la ReLU.

10. **`10_predicciones_prueba.png`: todo el 20% de prueba.** Las 38 imágenes de prueba con la clase real y la
    predicción de cada modelo (AE y CNN), en verde si acierta y en rojo si falla, más la exactitud total de cada modelo.

11. **`reporte.txt`:** precisión, recall y F1 por clase de ambos modelos.

## 6. Resultados obtenidos (20% de prueba, 38 imágenes)

| Modelo | Exactitud | Espiral (F1) | Lenticular (F1) | Elliptical (F1) |
|---|---|---|---|---|
| CNN | 68.4% | 0.86 | 0.29 | 0.36 |
| Autoencoder + clasificador | 63.2% | 0.84 | 0.33 | 0.00 |

**Interpretación:**
- Las **espirales** se reconocen bien porque tienen un rasgo muy distintivo, los brazos.
- **Elípticas y lenticulares se confunden mucho** porque visualmente son muy parecidas (núcleo brillante y
  halo difuso). Además hay muy pocas: solo 4 y 8 en la prueba.
- Con tan pocas imágenes de prueba, cada acierto o error mueve la exactitud unos 2.6 puntos, así que los
  porcentajes son poco estables.
- La CNN supervisada gana porque se entrena directamente para clasificar. El autoencoder aprende a reconstruir
  (brillo general y forma), que no es exactamente lo que distingue a una clase de otra.
- **Limitación:** el mismo 20% de prueba se usa como validación durante el entrenamiento (para EarlyStopping),
  así que el resultado es ligeramente optimista. Lo ideal sería una división 70/10/20.

## 7. Lo que te pido

**A) Para la presentación:**
- Propón una estructura de diapositivas (título, contenido y qué gráfica usar en cada una), en un orden
  lógico: problema → datos → autoencoder → CNN → resultados → conclusiones.
- Para cada diapositiva, escríbeme notas breves de lo que debo **decir** con mis propias palabras, en lenguaje sencillo.
- Sugiere cuáles gráficas son las más importantes y cuáles pueden ir como apoyo.
- Incluye una diapositiva de conclusiones y otra de limitaciones y mejoras posibles.

**B) Para estudiar:**
- Explícame paso a paso y con analogías estos conceptos: convolución, kernel/filtro, mapa de características,
  ReLU, max pooling, flatten, capa densa, softmax, dropout, autoencoder, encoder/decoder, código latente,
  de-noising, MSE, entropía cruzada, Adam, época, batch, sobreajuste, early stopping, transfer learning y
  fine-tuning, aumento de datos, pesos por clase, división estratificada, matriz de confusión, precisión, recall y F1.
- Explícame cómo se calcula el tamaño de salida de cada capa (por qué 64 → 32 → 16 → 8 y por qué el flatten da 4,096).
- Hazme un **cuestionario** de 15 preguntas (de fáciles a difíciles) como las que me haría el profesor,
  y después corrígeme las respuestas.
- Dame una lista de las **preguntas difíciles** que me podrían hacer (por ejemplo: "¿por qué el autoencoder
  clasifica peor?", "¿por qué no usaste imágenes a color?", "¿cómo mejorarías el modelo?") con buenas respuestas.

Ve paso a paso. Primero dame la estructura de la presentación y después pasamos al estudio de los conceptos.
