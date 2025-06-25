import os
import pydicom
import numpy as np
import cv2
from pathlib import Path

# Se determinan las rutas de los archivos
base_dir_train = "../../dataset/manifest/CBIS-DDSM/Mass_Training"
base_dir_val = "../../dataset/manifest/CBIS-DDSM/Mass_Test"
output_dir_train = Path("../../dataset/manifest/CBIS-DDSM/YOLO/images/train")
output_dir_val = Path("../../dataset/manifest/CBIS-DDSM/YOLO/images/val")

def create_main_folders(output_dir):
    """
    Función destinada a la confirmación de la existencia de la carpeta de destino.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

def process_image_no_resize(source_path, target_path):
    """
    Función destinada al procesamiento y guardado de la imagen sin recorte ni resize.
    Lee un archivo DICOM y lo guarda como .png sin redimensionar.
    """
    try:
        dicom_data = pydicom.dcmread(source_path)
        image_array = dicom_data.pixel_array

        # Convierte a 8 bits si fuera necesario
        if image_array.max() > 255:
            image_array = (image_array / image_array.max() * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)

        # Guarda la imagen en su tamaño original
        cv2.imwrite(str(target_path), image_array)
        print(f"Guardada sin redimensionar en: {target_path}")

    except Exception as e:
        print(f"Error al procesar {source_path}: {e}")

def reorganize_images(base_dir, output_dir):
    """
    Función principal destinada a la reorganización de imágenes con la misma estructura sin redimensionar.
    """
    create_main_folders(output_dir)

    for class_folder in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_folder)
        if os.path.isdir(class_path):  # Verifica si es un directorio (clase)
            print(f"Clase: {class_folder}")

            for first_level_folder in os.listdir(class_path):
                first_level_path = os.path.join(class_path, first_level_folder)
                if os.path.isdir(first_level_path):  # Verifica si es un directorio
                    print(f"Primer nivel: {first_level_folder}")

                    for second_level_folder in os.listdir(first_level_path):
                        second_level_path = os.path.join(first_level_path, second_level_folder)
                        if os.path.isdir(second_level_path):  # Verifica si es un directorio
                            print(f"Segundo nivel: {second_level_folder}")

                            for third_level_folder in os.listdir(second_level_path):
                                third_level_path = os.path.join(second_level_path, third_level_folder)
                                if os.path.isdir(third_level_path):  # Verifica si es un directorio
                                    words = third_level_folder.split("-")

                                    # full mammogram images
                                    if "full mammogram images" in words:
                                        # Procesar las imágenes "full mammogram images"
                                        for document in os.listdir(third_level_path):
                                            source_path = os.path.join(third_level_path, document)
                                            if os.path.isfile(source_path):
                                                base_name = f"{first_level_folder}_{second_level_folder}_{third_level_folder}"
                                                target_path = output_dir / f"{base_name}.png"
                                                process_image_no_resize(source_path, target_path)

                                    # ROI mask images
                                    elif "ROI mask images" in words:
                                        # Carpeta de salida para ROI en PNG (sin subcarpeta de clase)
                                        roi_output_folder = output_dir / "ROI mask images"

                                        # Crear las carpetas si no existen
                                        roi_output_folder.mkdir(parents=True, exist_ok=True)

                                        documents = sorted(os.listdir(third_level_path))  # Archivos .dcm, etc.
                                        for idx, document in enumerate(documents):
                                            source_path = os.path.join(third_level_path, document)
                                            if os.path.isfile(source_path):
                                                base_name = f"{first_level_folder}_{second_level_folder}_{third_level_folder}_{idx}"
                                                if idx == 1:
                                                    # Guardar la segunda imagen en "ROI mask images"
                                                    roi_target_path = roi_output_folder / f"{base_name}.png"
                                                    process_image_no_resize(source_path, roi_target_path)
                                                else: pass
                                    else: pass

if __name__ == "__main__":
    reorganize_images(base_dir_train, output_dir_train)
    reorganize_images(base_dir_val, output_dir_val)
