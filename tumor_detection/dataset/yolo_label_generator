import numpy as np
import cv2
from pathlib import Path

def generate_yolo_label_from_roi_png(roi_png_path, label_path, class_id=0):
    """
    Función destinada a la extracción del bounding box de la ROI mask y generación del .txt en formato YOLO dada una imagen.
    Procesa una máscara ROI en formato PNG para generar etiquetas YOLO.
    Asume que la máscara tiene fondo negro (0) y la ROI en blanco (>127).
    """
    try:
        mask_array = cv2.imread(str(roi_png_path), cv2.IMREAD_GRAYSCALE)

        if mask_array is None:
            print(f"No se pudo leer la imagen: {roi_png_path}")
            return

        height, width = mask_array.shape

        # Binariza
        _, thresh = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY)

        # Realiza operaciones morfológicas
        kernel = np.ones((5, 5), np.uint8)
        cleaned = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [c for c in contours if cv2.contourArea(c) > 50]

        if not contours:
            print(f"No se encontraron contornos en {roi_png_path.name}")
            return

        # Guarda la etiqueta YOLO
        with open(label_path, "w") as f:
            for c in contours:
                x, y, w_box, h_box = cv2.boundingRect(c)

                x_center = (x + w_box / 2) / width
                y_center = (y + h_box / 2) / height
                w_norm = w_box / width
                h_norm = h_box / height

                f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n")

        print(f"Etiqueta YOLO generada: {label_path}")

    except Exception as e:
        print(f"Error al procesar {roi_png_path.name}: {e}")


def process_roi_png_folder_in_place(roi_folder_path, class_id=0):
    """
    Función destinada a la extracción del bounding box de la ROI mask y generación del .txt en formato YOLO dada una carpeta.
    Recorre las imágenes .png de una carpeta procesando sus máscaras ROI para generar etiquetas YOLO.
    """
    roi_folder = Path(roi_folder_path)
    label_folder = roi_folder / "labels"
    label_folder.mkdir(exist_ok=True)

    for roi_png in sorted(roi_folder.glob("*.png")):
        base_name = roi_png.stem
        label_output_path = label_folder / f"{base_name}.txt"

        print(f"Procesando ROI PNG: {roi_png.name}")
        generate_yolo_label_from_roi_png(roi_png, label_output_path, class_id=class_id)

if __name__ == "__main__":
    roi_folder_path_val = "../../dataset/manifest/CBIS-DDSM/Mass_Test/ROI mask images/TODAS_JUNTAS(YOLO)"
    process_roi_png_folder_in_place(roi_folder_path_val, class_id=0)

    roi_folder_path_train = "../../dataset/manifest/CBIS-DDSM/Mass_Training/ROI mask images/TODAS_JUNTAS(YOLO)"
    process_roi_png_folder_in_place(roi_folder_path_train, class_id=0)