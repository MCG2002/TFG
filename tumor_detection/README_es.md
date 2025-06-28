# Detección de Tumores Mamarios con YOLOv8

Esta segunda sección del proyecto tiene como objetivo detectar automáticamente la localización de tumores en mamografías del dataset **CBIS-DDSM**, utilizando técnicas de visión por computadora y modelos de detección de objetos entrenados con **YOLOv8**.

Se automatiza el proceso completo: desde el preprocesamiento de imágenes y máscaras, hasta la generación de etiquetas y entrenamiento de modelos con distintas configuraciones (con y sin AdamW).

---

## Dataset

Este proyecto utiliza el **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)**, que incluye mamografías etiquetadas con características de los tumores.

---

## Estructura del Proyecto

```
tumor_detection/
│
├── dataset/
│   ├── folder_organizator.py         # Convierte DICOM a PNG sin redimensionar
│   ├── verify_label_png_names.py     # Verifica consistencia entre imágenes y etiquetas YOLO
│   ├── yolo_label_generator.py       # Genera bounding boxes YOLO desde máscaras ROI
│   └── yolo_label_grouper.py         # Agrupa múltiples etiquetas .txt por imagen base
│
├── models/
│   ├── graphic_evaluation/
│   │   ├── adamw_yolo_model.py       # Visualiza resultados del modelo YOLO con AdamW
│   │   └── base_yolo_model.py        # Visualiza resultados del modelo base
│   └── label_location_comparison/
│       ├── comparison_results/       # Contiene ejemplos de pruebas de los modelos YOLO
│       ├── mammo_yolo2/              # Carpeta con pesos entrenados
│       ├── original_labels.py        # Visualiza etiquetas manuales estilo YOLO
│       └── yolo_labels.py            # Visualiza predicciones de YOLO sobre mamografías
│
├── adamw_yolo_model.py               # Entrena modelo YOLO con AdamW y augmentations
└── base_yolo_model.py                # Entrena modelo YOLO estándar
```

---

## Flujo de Trabajo

### Conversión de Imágenes DICOM a PNG (sin resize)

```bash
python dataset/folder_organizator.py
```

Convierte las imágenes completas y las máscaras ROI del dataset CBIS-DDSM a PNG manteniendo el tamaño original.

---

### Generación de Etiquetas YOLO desde Máscaras ROI

```bash
python dataset/yolo_label_generator.py
```

Extrae los contornos de las máscaras (ROI) y crea archivos `.txt` en formato YOLO para cada imagen.

---

### Agrupación de Etiquetas por Imagen

```bash
python dataset/yolo_label_grouper.py
```

Agrupa archivos `.txt` correspondientes a una misma mamografía (identificada por prefijo), combinando múltiples regiones si es necesario.

---

### Verificación de Consistencia

```bash
python dataset/verify_label_png_names.py
```

Verifica que todos los archivos `.png` tengan su correspondiente etiqueta `.txt`.

---

### Entrenamiento del Modelo YOLO

#### Versión Base:

```bash
python base_yolo_model.py
```

#### Versión con Optimización AdamW y Aumento de datos:

```bash
python adamw_yolo_model.py
```

Entrena modelos YOLOv8 utilizando `ultralytics`, configurados para detectar tumores mamarios. El modelo se entrena con 100 épocas, imágenes 640x640 y probabilidad de flip horizontal activada.

---

## Visualización de Resultados

### Evaluación Gráfica del Entrenamiento:

Visualiza la evolución del entrenamiento en la carpeta `runs/detect/mammo_yolo*/results.png`.

```bash
python models/graphic_evaluation/admaw_yolo_model.py
# o
python models/graphic_evaluation/base_yolo_model.py
```

---

### Comparación de Detecciones

#### Etiquetas Manuales Estilo YOLO:

```bash
python models/label_location_comparison/original_labels.py
```

#### Predicción del Modelo YOLO:

```bash
python models/label_location_comparison/yolo_labels.py
```

Ambos scripts dibujan cajas delimitadoras al estilo YOLO para evaluar la precisión y exactitud del modelo.

#### Ejemplos de prueba para la Comparación de Detenciones

```bash
sh models/label_location_comparison/comparison_results/test.sh
```

---

## Requisitos

* Python 3.8+
* OpenCV
* NumPy
* pydicom
* matplotlib
* [Ultralytics](https://docs.ultralytics.com/) (`pip install ultralytics`)

Instalación rápida:

```bash
pip install opencv-python numpy matplotlib pydicom ultralytics
```
