import os
import pydicom
import numpy as np
import cv2
from PIL import Image
from pathlib import Path

# Se determinan las rutas de los archivos
base_dir_train = "../../dataset/manifest/CBIS-DDSM/Mass_Training"
base_dir_val = "../../dataset/manifest/CBIS-DDSM/Mass_Test"
output_dir_train = Path(base_dir_train)
output_dir_val = Path(base_dir_val)

# Se definen las carpetas en las que serán clasificadas las mamografías
main_folders = ["ROI mask images", "full mammogram images", "cropped images"]

def create_main_folders(output_dir):
    """
    Función destinada a la creación de carpetas principales si no existen previamente
    """
    for folder in main_folders:
        path = output_dir / folder
        if not path.exists():
            path.mkdir(parents=True)

def resize_with_padding(image, target_size, interpolation=cv2.INTER_AREA):
    """
    Función destinada a la redimensión de las imágenes añadiendo bordes negros
    """
    old_size = image.shape[:2]
    ratio = min(target_size[1] / old_size[0], target_size[0] / old_size[1])
    new_size = (int(old_size[1] * ratio), int(old_size[0] * ratio))

    resized_image = cv2.resize(image, new_size, interpolation=interpolation)

    delta_w = target_size[0] - new_size[0]
    delta_h = target_size[1] - new_size[1]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    # Agrega los bordes
    color = [0, 0, 0] # negro
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return padded_image

def preprocess_image(source_path, target_path, target_size):
    """
    Función destinada al procesamiento y guardado de imágenes.
    Lee un archivo DICOM, lo redimensiona añadiendo un borde negro y lo guarda como .png.
    """
    try:
        # Extrae los datos de píxeles
        dicom_data = pydicom.dcmread(source_path)
        image_array = dicom_data.pixel_array

        # Normaliza la imagen a 8 bits (0-255) si es necesario
        if image_array.max() > 255:
            image_array = (image_array / image_array.max() * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)

        # Redimensiona la imagen añadiendo borde negro
        resized_image = resize_with_padding(image_array, target_size, interpolation=cv2.INTER_AREA)

        # Guarda como imagen procesada
        cv2.imwrite(target_path, resized_image)
        print(f"Procesada y guardada en: {target_path}")
    except Exception as e:
        print(f"Error al procesar {source_path}: {e}")

def reorganize_images(base_dir, output_dir):
    """
    Función principal destinada a la reorganización y transformación de imágenes.
    """
    create_main_folders(output_dir)

    # Viaja entre las carpetas del dataset hasta alcanzar los documentos fundamentales para el estudio
    for class_folder in os.listdir(base_dir):
        class_path = os.path.join(base_dir, class_folder)
        if os.path.isdir(class_path):
            print(f"Clase: {class_folder}")

            for first_level_folder in os.listdir(class_path):
                first_level_path = os.path.join(class_path, first_level_folder)
                if os.path.isdir(first_level_path):
                    print(f"Primer nivel: {first_level_folder}")

                    for second_level_folder in os.listdir(first_level_path):
                        second_level_path = os.path.join(first_level_path, second_level_folder)
                        if os.path.isdir(second_level_path):
                            print(f"Segundo nivel: {second_level_folder}")

                            for third_level_folder in os.listdir(second_level_path):
                                third_level_path = os.path.join(second_level_path, third_level_folder)
                                if os.path.isdir(third_level_path):
                                    words = third_level_folder.split("-")

                                    # Verifica el tipo de carpeta y procesa los documentos
                                    if "ROI mask images" in words:
                                        roi_output_folder = output_dir / "ROI mask images" / class_folder
                                        cropped_output_folder = output_dir / "cropped images" / class_folder
                                        target_size_roi = (800, 1350)
                                        target_size_cropped = (550, 550)

                                        # Crea las carpetas de salida si no existen
                                        if not roi_output_folder.exists():
                                            roi_output_folder.mkdir(parents=True)
                                        if not cropped_output_folder.exists():
                                            cropped_output_folder.mkdir(parents=True)

                                        documents = sorted(os.listdir(third_level_path))
                                        for idx, document in enumerate(documents):
                                            source_path = os.path.join(third_level_path, document)
                                            if os.path.isfile(source_path):
                                                if idx == 0:  # La carpeta asignada a la primera imagen es "cropped images"
                                                    target_path = cropped_output_folder / f"{class_folder}_{first_level_folder}_{second_level_folder}_{document.replace('.dcm', '.png')}"
                                                    preprocess_image(source_path, target_path, target_size_cropped)
                                                elif idx == 1:  # La carpeta asignada a la primera imagen es "ROI mask images"
                                                    target_path = roi_output_folder / f"{class_folder}_{first_level_folder}_{second_level_folder}_{document.replace('.dcm', '.png')}"
                                                    preprocess_image(source_path, target_path, target_size_roi)
                                    elif "full mammogram images" in words:
                                        output_folder = output_dir / "full mammogram images" / class_folder
                                        target_size = (800, 1350)

                                        if not output_folder.exists():
                                            output_folder.mkdir(parents=True)

                                        for document in os.listdir(third_level_path):
                                            source_path = os.path.join(third_level_path, document)
                                            if os.path.isfile(source_path):
                                                target_path = output_folder / f"{class_folder}_{first_level_folder}_{second_level_folder}_{document.replace('.dcm', '.png')}"
                                                preprocess_image(source_path, target_path, target_size)
                                    elif "cropped images" in words:
                                        output_folder = output_dir / "cropped images" / class_folder
                                        target_size = (550, 550)

                                        if not output_folder.exists():
                                            output_folder.mkdir(parents=True)

                                        for document in os.listdir(third_level_path):
                                            source_path = os.path.join(third_level_path, document)
                                            if os.path.isfile(source_path):
                                                target_path = output_folder / f"{class_folder}_{first_level_folder}_{second_level_folder}_{document.replace('.dcm', '.png')}"
                                                preprocess_image(source_path, target_path, target_size)

if __name__ == "__main__":
    reorganize_images(base_dir_train, output_dir_train)
    reorganize_images(base_dir_val, output_dir_val)
