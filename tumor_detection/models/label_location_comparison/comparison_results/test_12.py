from tumor_detection.models.label_location_comparison.original_labels import draw_yolo_labels
from tumor_detection.models.label_location_comparison.yolo_labels import yolo_predict

image_path = "../../../dataset/manifest/CBIS-DDSM/YOLO/images/val/Mass_Test_P_00126_RIGHT_CC.png"
label_path = "../../../dataset/manifest/CBIS-DDSM/YOLO/labels/val/Mass_Test_P_00126_RIGHT_CC.txt"
draw_yolo_labels(image_path, label_path)
yolo_predict(image_path)