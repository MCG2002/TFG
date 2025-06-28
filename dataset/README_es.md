# Dataset

Este proyecto emplea el **CBIS-DDSM: Curated Breast Imaging Subset of DDSM**, desarrollado por Sawyer-Lee et al. (2016). Es una versión actualizada y curada del clásico **DDSM** (Digital Database for Screening Mammography), con mejoras en la calidad de anotaciones, accesibilidad y segmentación.

El dataset incluye **10,237 mamografías** de **1,566 pacientes**, y clasifica las imágenes en tres tipos principales:

* **Mamografías completas**
* **Imágenes recortadas** que muestran la región sospechosa
* **Máscaras segmentadas (ROI)** que indican la localización del tumor

| Tipo de Imagen      | Número de muestras |
| ------------------- | ------------------ |
| Mamografía completa | 2,857              |
| Recortada           | 3,567              |
| ROI                 | 3,247              |
| No válidas          | 566                |

---

## Archivos Adicionales

El dataset incluye dos archivos CSV:

* **Calcificaciones** (no se utilizan en este proyecto)
* **Masas tumorales** (sí se utilizan)

Estos archivos contienen información detallada sobre:

* **Patología**: Las masas mamarias se clasifican en tres grandes grupos:

  * **Malignas**: con crecimiento celular descontrolado y características propias del cáncer.
  * **Benignas**: no cancerosas, generalmente de crecimiento lento y con bordes bien definidos.
  * **Benignas sin nueva evaluación**: tumores benignos estables que no requieren seguimiento adicional.

* **Categoría BI-RADS**: Sistema estandarizado utilizado por los radiólogos para describir los hallazgos en mamografías. Las categorías van del 0 al 6:

  * **0**: Resultado incompleto, se requiere más evaluación.
  * **1**: Resultado negativo.
  * **2**: Hallazgo benigno.
  * **3**: Probablemente benigno, se recomienda seguimiento.
  * **4**: Anomalía sospechosa, se sugiere biopsia.
  * **5**: Alta sospecha de malignidad.
  * **6**: Cáncer confirmado por biopsia.

* **Forma de la masa**:

  * **Espiculada**, **circunscrita**, **poco definida**, **oculta** y **microlobulada**.

* **Márgenes**:

  * **Circunscritos**, **espiculados**, **poco definidos**, **ocultos**, **microlobulados** y combinaciones de estos.

* **Densidad mamaria**: Se distinguen cuatro tipos:

  * (1) Grasa
  * (2) Fibroglandular dispersa
  * (3) Densidad heterogénea
  * (4) Extremadamente densa

  Se considera densidad **baja** para valores 1 y 2, y **alta** para 3 y 4.

---

## Clasificaciones Derivadas

A partir de los CSV, se definió una **clasificación extendida** más precisa que la original, incorporando tanto la naturaleza del tumor (benigno o maligno) como un estimado del estadio clínico.

Se identifican **11 clases finales**:

* Benigno Estable
* Benigno
* Benigno Peligroso
* Maligno Estadio 0-1
* Maligno Estadio 1-2
* Maligno Estadio 2
* Maligno Estadio 2-3
* Maligno Estadio 3
* Maligno Estadio 3-4
* Maligno Estadio 4
* Diagnóstico Incompleto *(excluido del entrenamiento)*

### Distribución por Clase

| Clase                  | Porcentaje |
| ---------------------- | ---------- |
| Benigno Estable        | 7.89%      |
| Benigno                | 20.94%     |
| Benigno Peligroso      | 16.08%     |
| Maligno Estadio 3-4    | 23.07%     |
| Maligno Estadio 4      | 22.23%     |
| Diagnóstico Incompleto | 9.79%      |

Las clases restantes no están representadas en el conjunto de entrenamiento.

---

## Preprocesamiento

El dataset es preprocesado para asegurar uniformidad, reducir la carga computacional y mejorar el rendimiento del modelo.

### Técnicas Aplicadas

* **Organización por clases**: Las imágenes se agrupan en tres carpetas base — `full mammogram images`, `cropped images` y `ROI mask images` —, cada una subdividida en las 11 clases anteriores.

* **Redimensionado con padding**:

  * Mamografías completas y máscaras ROI: `800x1350 px`
  * Imágenes recortadas: `550x550 px`
  * Se preserva la proporción original y se añade relleno negro si es necesario.

* **Normalización**: Los valores de los píxeles se escalan al rango `[0, 1]`.

* **Conversión de formato**: Todas las imágenes se convierten de DICOM a PNG para facilitar su uso en modelos de deep learning.

---
