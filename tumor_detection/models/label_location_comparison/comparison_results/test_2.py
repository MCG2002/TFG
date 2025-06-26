from tumor_detection.models.label_location_comparison.original_labels import draw_yolo_labels
from tumor_detection.models.label_location_comparison.yolo_labels import yolo_predict

image_path = "../../../dataset/manifest/CBIS-DDSM/YOLO/images/train/Mass_Training_P_00914_LEFT_CC.png"
label_path = "../../../dataset/manifest/CBIS-DDSM/YOLO/labels/train/Mass_Training_P_00914_LEFT_CC.txt"
draw_yolo_labels(image_path, label_path)
yolo_predict(image_path)