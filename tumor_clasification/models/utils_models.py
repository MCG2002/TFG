import matplotlib.pyplot as plt
import tensorflow as tf


def plot_loss_acc(history):
    """
    Función destinada a la generación de dos gráficas que muestran la evolución del entrenamiento y validación de un
    modelo de aprendizaje automático a lo largo de las épocas extrayendo la información del historial de información.
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Gráficas de la precisión y la pérdida del entrenamiento y la validación')
    for i, (data, label) in enumerate(zip([(acc, val_acc), (loss, val_loss)], ["Precisión", "Pérdida"])):
        ax[i].plot(epochs, data[0], 'r', label="Entrenamiento " + label)
        ax[i].plot(epochs, data[1], 'b', label="Validación " + label)
        ax[i].legend()
        ax[i].set_xlabel('epochs')
    plt.show()


def create_data_augmentation():
    """
    Función destinada a la especificación de los parámetros empleados en el Aumento de Datos.
    """
    return tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.3),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomTranslation(0.2, 0.2)
    ])


class EarlyStoppingCallback(tf.keras.callbacks.Callback):
    """
    Esta clase define una callback personalizada de Detección Temprana para Keras. 
    Su función es detener automáticamente el entrenamiento del modelo si se cumplen las condiciones determinadas en la
    función on_epoch_end.
    """
    def on_epoch_end(self, logs=None):
        """
        Función destinada a la detección automática del entrenamiento del modelo si se cumplen simultáneamente las
        siguientes condiciones:
          - La excatitud del entrenamiento es mayor o igual a 0.95.
          - La exactitud de la validación es mayor o igual a 0.90.
        """
        if logs["accuracy"] >= 0.95 and logs["val_accuracy"] >= 0.90:
            self.model.stop_training = True
            print("\nSe alcanzaron los criterios de Detección Temprana.")