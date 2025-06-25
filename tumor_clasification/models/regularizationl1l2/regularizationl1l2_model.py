import tensorflow as tf
from tensorflow.keras.regularizers import L1L2
from tumor_clasification.models.utils_models import plot_loss_acc
from tumor_clasification.dataset.image_procesor import load_train_images, load_val_images

full_train_images, full_train_labels, class_names, cropped_train_images, cropped_train_labels = load_train_images()
full_val_images, full_val_labels, cropped_val_images, cropped_val_labels = load_val_images()

def create_multimodal_model_with_l1l2(input_shape_full, input_shape_cropped, num_classes, l1_value, l2_value):
    """
    Función destinada a la creacción de un nuevo modelo multimodal que incluye las técnicas denominadas como Regularización 
    L1 y L2 a partir del modelo multimodal anterior sin Aumento de Datos incluido.
    """
    # Submodelo para imágenes completas
    input_full = tf.keras.Input(shape=input_shape_full, name="full_mammogram_input")
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
    x_cropped = tf.keras.layers.Rescaling(1.0 / 255)(input_cropped)
    x_cropped = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(x_cropped)
    x_cropped = tf.keras.layers.BatchNormalization()(x_cropped)
    x_cropped = tf.keras.layers.MaxPooling2D((2, 2))(x_cropped)
    x_cropped = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x_cropped)
    x_cropped = tf.keras.layers.BatchNormalization()(x_cropped)
    x_cropped = tf.keras.layers.MaxPooling2D((2, 2))(x_cropped)
    x_cropped = tf.keras.layers.Flatten()(x_cropped)

    # Concatenación y capas finales con Regularización L1 y L2
    concatenated = tf.keras.layers.Concatenate()([x_full, x_cropped])
    x = tf.keras.layers.Dense(128, activation="relu", kernel_regularizer=L1L2(l1=l1_value, l2=l2_value))(concatenated)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", kernel_regularizer=L1L2(l1=l1_value, l2=l2_value))(x)

    model = tf.keras.Model(inputs=[input_full, input_cropped], outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


model_with_l1l2 = create_multimodal_model_with_l1l2(
    input_shape_full=(800, 1350, 1),
    input_shape_cropped=(550, 550, 1),
    num_classes=len(class_names),
    l1_value=0.01,
    l2_value=0.01
)


# Se almacena toda la información del proceso de entrenamiento y validación sin Aumento de Datos incluido
history_with_l1l2 = model_with_l1l2.fit(
    {"full_mammogram_input": full_train_images, "cropped_image_input": cropped_train_images},
    full_train_labels,
    validation_data=(
        {"full_mammogram_input": full_val_images, "cropped_image_input": cropped_val_images},
        full_val_labels
    ),
    epochs=10,
    batch_size=12
)


plot_loss_acc(history_with_l1l2)