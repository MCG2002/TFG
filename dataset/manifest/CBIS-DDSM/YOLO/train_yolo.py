from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")

    model.train(
        data="mammo.yaml",    
        epochs=50,            
        imgsz=640,            
        batch=8,             
        name="mammo_yolo",    # carpeta de resultados runs/detect/mammo_yolo
    )

if __name__ == "__main__":
    main()
