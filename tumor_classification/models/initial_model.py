import tensorflow as tf
from tumor_classification.models.utils_models import plot_loss_acc
from tumor_classification.dataset.image_procesor import load_train_images, load_val_images

full_train_images, full_train_labels, class_names, cropped_train_images, cropped_train_labels = load_train_images()
full_val_images, full_val_labels, cropped_val_images, cropped_val_labels = load_val_images()

def create_multimodal_model(input_shape_full, input_shape_cropped, num_classes):
    """
    Función destinada a la creacción del modelo multimodal empleado en este TFG.

    La arquitectura se compone de dos submodelos independientes, uno para las mamografías completas y otro para las 
    recortadas donde se muestra la localización ampliada del tumor. Ambos submodelos siguen una estructura similar de
    capas convolucionales pero funcionan como flujos de procesamiento separados hasta que sus salidas son combinadas
    con el fin de optimizar la eficiencia del entrenamiento y reducir la sobrecarga computacional.
    """
    # Submodelo para imágenes completas
    input_full = tf.keras.Input(shape=input_shape_full, name="full_mammogram_input")
    x_full = tf.keras.layers.Rescaling(1.0 / 255)(input_full)
    x_full = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(x_full)
    x_full = tf.keras.layers.MaxPooling2D((2, 2))(x_full)
    x_full = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x_full)
    x_full = tf.keras.layers.MaxPooling2D((2, 2))(x_full)
    x_full = tf.keras.layers.Flatten()(x_full)

    # Submodelo para imágenes recortadas
    input_cropped = tf.keras.Input(shape=input_shape_cropped, name="cropped_image_input")
    x_cropped = tf.keras.layers.Rescaling(1.0 / 255)(input_cropped)
    x_cropped = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(x_cropped)
    x_cropped = tf.keras.layers.MaxPooling2D((2, 2))(x_cropped)
    x_cropped = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(x_cropped)
    x_cropped = tf.keras.layers.MaxPooling2D((2, 2))(x_cropped)
    x_cropped = tf.keras.layers.Flatten()(x_cropped)

    # Concatenación y capas finales
    concatenated = tf.keras.layers.Concatenate()([x_full, x_cropped])
    x = tf.keras.layers.Dense(128, activation="relu")(concatenated)
    output = tf.keras.layers.Dense(num_classes, activation="softmax", name="class_output")(x)

    model = tf.keras.Model(inputs=[input_full, input_cropped], outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


model_multimodal = create_multimodal_model(
    input_shape_full=(800, 1350, 1), 
    input_shape_cropped=(550, 550, 1), 
    num_classes=len(class_names)
)

# Se almacena toda la información del proceso de entrenamiento y validación
history = model_multimodal.fit(
    {"full_mammogram_input": full_train_images, "cropped_image_input": cropped_train_images},
    full_train_labels,
    validation_data=(
        {"full_mammogram_input": full_val_images, "cropped_image_input": cropped_val_images},
        full_val_labels
    ),
    epochs=10,
    batch_size=12
)


plot_loss_acc(history)