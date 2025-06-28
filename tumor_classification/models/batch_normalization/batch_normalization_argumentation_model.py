import tensorflow as tf
from tumor_classification.models.utils_models import plot_loss_acc, create_data_augmentation
from tumor_classification.dataset.image_procesor import load_train_images, load_val_images

full_train_images, full_train_labels, class_names, cropped_train_images, cropped_train_labels = load_train_images()
full_val_images, full_val_labels, cropped_val_images, cropped_val_labels = load_val_images()

def create_multimodal_model_with_bn_and_aug(input_shape_full, input_shape_cropped, num_classes):
    """
    Función destinada a la creacción de un nuevo modelo multimodal que incluye la técnica denominada como Normalización
    por Lotes a partir del modelo multimodal anterior con Aumento de Datos incluido.
    """
    data_augmentation = create_data_augmentation()

    # Submodelo para imágenes completas
    input_full = tf.keras.Input(shape=input_shape_full, name="full_mammogram_input")
    x_full = data_augmentation(input_full)
    x_full = tf.keras.layers.Rescaling(1.0 / 255)(input_full)
    x_full = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(x_full)
    x_full = tf.keras.layers.BatchNormalization()(x_full)
    x_full = tf.keras.layers.MaxPooling2D((2, 2))(x_full)
    x_full = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x_full)
    x_full = tf.keras.layers.BatchNormalization()(x_full)
    x_full = tf.keras.layers.MaxPooling2D((2, 2))(x_full)
    x_full = tf.keras.layers.Flatten()(x_full)

    # Submodelo para imágenes recortadas
    input_cropped = tf.keras.Input(shape=input_shape_cropped, name="cropped_image_input")
    x_cropped = data_augmentation(input_cropped)
    x_cropped = tf.keras.layers.Rescaling(1.0 / 255)(input_cropped)
    x_cropped = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(x_cropped)
    x_cropped = tf.keras.layers.BatchNormalization()(x_cropped)
    x_cropped = tf.keras.layers.MaxPooling2D((2, 2))(x_cropped)
    x_cropped = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x_cropped)
    x_cropped = tf.keras.layers.BatchNormalization()(x_cropped)
    x_cropped = tf.keras.layers.MaxPooling2D((2, 2))(x_cropped)
    x_cropped = tf.keras.layers.Flatten()(x_cropped)

    # Concatenación y capas finales con Batch Normalization
    concatenated = tf.keras.layers.Concatenate()([x_full, x_cropped])
    x = tf.keras.layers.Dense(128, activation="relu")(concatenated)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_output")(x)

    model = tf.keras.Model(inputs=[input_full, input_cropped], outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


model_with_bn_and_aug = create_multimodal_model_with_bn_and_aug(
    input_shape_full=(800, 1350, 1),
    input_shape_cropped=(550, 550, 1),
    num_classes=len(class_names)
)


history_with_bn_and_aug = model_with_bn_and_aug.fit(
    {"full_mammogram_input": full_train_images, "cropped_image_input": cropped_train_images},
    full_train_labels,
    validation_data=(
        {"full_mammogram_input": full_val_images, "cropped_image_input": cropped_val_images},
        full_val_labels
    ),
    epochs=10,
    batch_size=12
)


plot_loss_acc(history_with_bn_and_aug)