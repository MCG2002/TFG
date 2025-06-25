import os
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras.utils import img_to_array

mass_train_dir = "../../dataset/manifest/CBIS-DDSM/Mass_Training"
mass_test_dir = "../../dataset/manifest/CBIS-DDSM/Mass_Test"

target_size_full = (800, 1350)
target_size_cropped = (550, 550)
batch_size = 16

def load_images_from_mass_structure(base_dir, image_type, target_size):
    """
    Función destinada a la carga de imágenes PNG desde los directorios donde se encuentran los datasets de entrenamiento
    y validación.
    """
    images = []
    labels = []
    class_dir = os.path.join(base_dir, image_type)
    class_names = sorted(os.listdir(class_dir))

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(class_dir, class_name)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                if file_name.endswith(".png"):
                    img_path = os.path.join(class_path, file_name)
                    try:
                        with Image.open(img_path) as img:
                            if img.mode != "L":
                                img = img.convert("L") 
                            img = img.resize(target_size, Image.Resampling.LANCZOS)
                            image_array = img_to_array(img)
                            images.append(image_array)
                            labels.append(label)
                        print(f"Imagen {img_path} procesada")
                    except Exception as e:
                        print(f"Error al procesar la imagen {img_path}: {e}")

    return np.array(images), np.array(labels), class_names


def load_train_images():

    full_train_images, full_train_labels, class_names = load_images_from_mass_structure(
        mass_train_dir, "full mammogram images", target_size=target_size_full
    )
    cropped_train_images, cropped_train_labels, _ = load_images_from_mass_structure(
        mass_train_dir, "cropped images", target_size=target_size_cropped
    )

    dimension_corrector(full_train_images, cropped_train_images, None, None)

    return full_train_images, full_train_labels, class_names, cropped_train_images, cropped_train_labels

def load_val_images():

    full_val_images, full_val_labels, _ = load_images_from_mass_structure(
        mass_test_dir, "full mammogram images", target_size=target_size_full
    )
    cropped_val_images, cropped_val_labels, _ = load_images_from_mass_structure(
        mass_test_dir, "cropped images", target_size=target_size_cropped
    )

    dimension_corrector(None, None, full_val_images, cropped_val_images)

    return full_val_images, full_val_labels, cropped_val_images, cropped_val_labels


def dimension_corrector(full_train_images, cropped_train_images, full_val_images, cropped_val_images):

    # Se verifican las dimensiones iniciales
    print("Formas iniciales:")
    print("Full train:", full_train_images.shape)
    print("Cropped train:", cropped_train_images.shape)
    print("Full val:", full_val_images.shape)
    print("Cropped val:", cropped_val_images.shape)

    # Se corrigen las dimensiones si están invertidas
    full_train_images = np.transpose(full_train_images, (0, 2, 1, 3))
    cropped_train_images = np.transpose(cropped_train_images, (0, 2, 1, 3))
    full_val_images = np.transpose(full_val_images, (0, 2, 1, 3))
    cropped_val_images = np.transpose(cropped_val_images, (0, 2, 1, 3))

    # Se verifican las dimensiones después de la corrección
    print("Formas corregidas:")
    print("Full train:", full_train_images.shape)
    print("Cropped train:", cropped_train_images.shape)
    print("Full val:", full_val_images.shape)
    print("Cropped val:", cropped_val_images.shape)