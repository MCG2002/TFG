import matplotlib.pyplot as plt

def yolo_predict(image_path):
    """
    Función destinada a la visualización de las mamografías destacando la localización de los tumores existentes 
    según las predicciones obtenidas por YOLO.
    """
    # Carga el modelo entrenado
    model = YOLO("runs/detect/mammo_yolo2/weights/best.pt")

    # Predecice la posición de los tumores con YOLO
    results = model.predict(source=image_path)

    # Obtiene el resultado
    res = results[0]

    # Dibujan las cajas en la imagen
    img_annotated = res.plot()

    # Muestra el resultado
    plt.imshow(img_annotated)
    plt.axis('off')
    plt.show()