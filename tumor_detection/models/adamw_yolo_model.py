from ultralytics import YOLO

def yolo_adamw_flip():
    """
    Función destinada al entrenamiento de un modelo YOLOv8 incluyendo el uso del optimizador AdamW y la técnica denominada
    como Aumento de Datos con el fin de localizar con la mayor exactitud posible la localización de los tumores dado
    el dataset de mamografías.
    """
    # Carga el modelo base "yolov8n.pt" y lo entrena usando la configuración definida en el archivo YAML
    model = YOLO("yolov8n.pt")

    # Entrena durante 100 épocas con imágenes de tamaño 640x640 y un batch de 8
    model.train(
        data=r"C:\Users\mcamp\OneDrive\Documentos\TFG_P2\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM\YOLO\mammo.yaml",    
        epochs=100,
        imgsz=640,
        batch=8,
        name="mammo_yolo_adamw_flip",
        augment=True,    # Habilita el augmentation
        fliplr=0.5,      # Probabilidad de flip horizontal = 50%
        flipud=0.0,      # Probabilidad de flip vertical = 0% (desactivado)
        optimizer="AdamW",
        plots=True 
    )


if __name__ == "__main__":
    yolo_adamw_flip()