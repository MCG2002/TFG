from tumor_clasification.models.utils_models import EarlyStoppingCallback, plot_loss_acc
from tumor_clasification.models.regularizationl1l2.regularizationl1l2_model import model_with_l1l2
from tumor_clasification.dataset.image_procesor import load_train_images, load_val_images

full_train_images, full_train_labels, class_names, cropped_train_images, cropped_train_labels = load_train_images()
full_val_images, full_val_labels, cropped_val_images, cropped_val_labels = load_val_images()

# Se almacena toda la información del proceso de entrenamiento y validación sin Aumento de Datos incluido
history_with_early_stopping = model_with_l1l2.fit(
    {"full_mammogram_input": full_train_images, "cropped_image_input": cropped_train_images},
    full_train_labels,
    validation_data=(
        {"full_mammogram_input": full_val_images, "cropped_image_input": cropped_val_images},
        full_val_labels
    ),
    epochs=10,
    batch_size=12,
    callbacks=[EarlyStoppingCallback()]
)


plot_loss_acc(history_with_early_stopping)