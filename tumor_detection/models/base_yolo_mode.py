from ultralytics import YOLO

def yolo():
    """
    Función destinada al entrenamiento de un modelo YOLOv8 con el fin de localizar con la mayor exactitud posible 
    la localización de los tumores dado el dataset de mamografías.
    """
    # Carga el modelo base "yolov8n.pt"
    model = YOLO("yolov8n.pt")

    # Entrena durante 100 épocas con imágenes de tamaño 640x640 y un batch de 8
    model.train(
        data=r"C:\Users\mcamp\OneDrive\Documentos\TFG_P2\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM\YOLO\mammo.yaml",    
        epochs=100,
        imgsz=640,
        batch=8,
        name="mammo_yolo",  # Guarda el experimento bajo el nombre "mammo_yolo"
        plots=True          # Genera gráficas del entrenamiento.
    )


if __name__ == "__main__":
    yolo()