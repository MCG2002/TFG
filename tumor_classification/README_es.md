# Clasificación de Tumores en Mamografías

La primera sección de este proyecto tiene como objetivo clasificar tumores mamarios mediante el uso de redes neuronales convolucionales aplicadas a imágenes médicas DICOM del dataset CBIS-DDSM. La arquitectura emplea un enfoque **multimodal**, combinando imágenes completas y recortadas de las mamografías para mejorar el rendimiento del modelo.

---

## Dataset

Este proyecto utiliza el **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)**, que incluye mamografías etiquetadas con características de los tumores.

---

## Estructura del Proyecto

```
tumor_classification/
│
├── dataset/
│   ├── folder_organizator.py       # Organiza y preprocesa las imágenes DICOM
│   └── image_procesor.py           # Carga y normaliza las imágenes para entrenamiento/validación
│
├── models/
│   ├── augmentation/               # Modelos con técnicas de aumento de datos
│   ├── batch_normalization/        # Modelos con normalización por lotes
│   ├── dropout/                    # Modelos con regularización por Dropout
│   ├── early_stopping/             # Modelos entrenados con Early Stopping
│   ├── regularizationl1/           # Regularización L1
│   ├── regularizationl2/           # Regularización L2
│   ├── regularizationl1l2/         # Regularización combinada L1+L2
│   ├── initial_model.py            # Modelo base multimodal
│   └── utils_models.py             # Funciones de apoyo (visualización, métricas, etc.)
```

---

## Preprocesamiento de Datos

El archivo `folder_organizator.py` organiza las imágenes DICOM en carpetas según su tipo (imagen completa, ROI, recortada) y las convierte a formato PNG. Las imágenes se redimensionan manteniendo la relación de aspecto y se les añade padding negro.

```bash
python dataset/folder_organizator.py
```

Luego, las imágenes son cargadas y preparadas para el entrenamiento mediante `image_procesor.py`, que se encarga de:

* Convertir imágenes a escala de grises.
* Ajustar las dimensiones y normalizar.
* Retornar los tensores listos para el modelo.

---

## Modelo Multimodal

El archivo `initial_model.py` define el modelo base inicial que toma dos entradas:

* **Mamografía completa** (800x1350)
* **Región recortada** (550x550)

Ambos caminos son redes convolucionales paralelas cuyas salidas se concatenan para generar una predicción final. El modelo se entrena durante 10 épocas con `categorical_crossentropy` y `Adam`.

---

## Técnicas de Optimización

Se incluyen varias versiones del modelo base con diferentes técnicas de mejora:

| Carpeta                | Técnica Aplicada                          |
| ---------------------- | ----------------------------------------- |
| `augmentation/`        | Aumento de datos (rotación, flip, etc.)   |
| `batch_normalization/` | Normalización por lotes                   |
| `dropout/`             | Dropout en capas densas y convolucionales |
| `early_stopping/`      | Entrenamiento con detención temprana      |
| `regularizationl1/`    | Regularización L1                         |
| `regularizationl2/`    | Regularización L2                         |
| `regularizationl1l2/`  | Regularización combinada L1 y L2          |

---

## Resultados y Métricas

Durante el entrenamiento, se grafican la **pérdida** y **precisión** en entrenamiento/validación mediante una función importada de `utils_models.py`.

---

## Ejecución

1. Preprocesar imágenes DICOM:

   ```bash
   python dataset/folder_organizator.py
   ```

2. Cargar y normalizar imágenes:

   ```bash
   python dataset/image_procesor.py
   ```

3. Entrenar modelo base:

   ```bash
   python models/initial_model.py
   ```

4. Probar variantes con regularización, dropout, etc., ejecutando el modelo correspondiente.

---

## Requisitos

* Python 3.8+
* TensorFlow 2.x
* Pillow
* NumPy
* OpenCV
* pydicom

Instalación:

```bash
pip install tensorflow pillow numpy opencv-python pydicom
```
