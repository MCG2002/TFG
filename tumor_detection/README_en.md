# Breast Tumor Detection with YOLOv8

This second part of the project aims to automatically detect the location of tumors in mammograms from the **CBIS-DDSM** dataset, using computer vision techniques and object detection models trained with **YOLOv8**.

The entire pipeline is automated — from preprocessing images and masks to generating labels and training models with different configurations (with and without AdamW optimization).

---

## Dataset

This project uses the **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)**, which includes mammograms labeled with tumor characteristics.

---

## Project Structure

```
tumor_detection/
│
├── dataset/
│   ├── folder_organizator.py         # Converts DICOM to PNG without resizing
│   ├── verify_label_png_names.py     # Verifies consistency between images and YOLO labels
│   ├── yolo_label_generator.py       # Generates YOLO bounding boxes from ROI masks
│   └── yolo_label_grouper.py         # Merges multiple .txt labels per base image
│
├── models/
│   ├── graphic_evaluation/
│   │   ├── adamw_yolo_model.py       # Visualizes results of YOLO model with AdamW
│   │   └── base_yolo_model.py        # Visualizes results of the base YOLO model
│   └── label_location_comparison/
│       ├── comparison_results/       # Contains test examples of YOLO models
│       ├── mammo_yolo2/              # Folder with trained weights
│       ├── original_labels.py        # Displays manual YOLO-style labels
│       └── yolo_labels.py            # Displays YOLO predictions on mammograms
│
├── adamw_yolo_model.py               # Trains YOLO model with AdamW and augmentations
└── base_yolo_model.py                # Trains the standard YOLO model
```

---

## Workflow

### DICOM to PNG Conversion (No Resize)

```bash
python dataset/folder_organizator.py
```

Converts full mammogram images and ROI masks from the CBIS-DDSM dataset into PNG format, preserving the original size.

---

### YOLO Label Generation from ROI Masks

```bash
python dataset/yolo_label_generator.py
```

Extracts contours from ROI masks and creates YOLO-format `.txt` label files for each image.

---

### Grouping Labels per Image

```bash
python dataset/yolo_label_grouper.py
```

Groups `.txt` label files associated with the same mammogram (identified by prefix), combining multiple regions if needed.

---

### Label-Image Consistency Check

```bash
python dataset/verify_label_png_names.py
```

Ensures that every `.png` image has a corresponding `.txt` YOLO label file.

---

### YOLO Model Training

#### Base Version:

```bash
python base_yolo_model.py
```

#### Version with AdamW Optimization and Data Augmentation:

```bash
python adamw_yolo_model.py
```

Trains YOLOv8 models using the `ultralytics` package, configured to detect breast tumors. The model trains for 100 epochs on 640x640 images with horizontal flip augmentation enabled.

---

## Result Visualization

### Training Performance Evaluation:

View training progress in `runs/detect/mammo_yolo*/results.png`.

```bash
python models/graphic_evaluation/adamw_yolo_model.py
# or
python models/graphic_evaluation/base_yolo_model.py
```

---

### Detection Comparison

#### Manual YOLO-style Labels:

```bash
python models/label_location_comparison/original_labels.py
```

#### YOLO Model Predictions:

```bash
python models/label_location_comparison/yolo_labels.py
```

Both scripts draw YOLO-style bounding boxes to assess the model’s accuracy and precision.

#### Example Comparisons:

```bash
sh models/label_location_comparison/comparison_results/test.sh
```

---

## Requirements

* Python 3.8+
* OpenCV
* NumPy
* pydicom
* matplotlib
* [Ultralytics](https://docs.ultralytics.com/) (`pip install ultralytics`)

Quick installation:

```bash
pip install opencv-python numpy matplotlib pydicom ultralytics
```
