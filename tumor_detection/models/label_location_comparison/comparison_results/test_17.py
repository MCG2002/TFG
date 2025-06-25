from tumor_detection.models.label_location_comparison.original_labels import draw_yolo_labels
from tumor_detection.models.label_location_comparison.yolo_labels import yolo_predict

image_path = r"C:\Users\mcamp\OneDrive\Documentos\TFG_P2\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM\YOLO\images\val\Mass_Test_P_01251_LEFT_MLO.png"
label_path = r"c:\Users\mcamp\OneDrive\Documentos\TFG_P2\manifest-ZkhPvrLo5216730872708713142\CBIS-DDSM\YOLO\labels\val\Mass_Test_P_01251_LEFT_MLO.txt"
draw_yolo_labels(image_path, label_path)
yolo_predict(image_path)