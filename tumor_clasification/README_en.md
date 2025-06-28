# Tumor Classification in Mammograms

The first phase of this project focuses on classifying breast tumors using convolutional neural networks applied to medical DICOM images from the CBIS-DDSM dataset. The architecture adopts a **multimodal** approach, combining full and cropped mammogram images to enhance model performance.

---

## Dataset

This project uses the **CBIS-DDSM (Curated Breast Imaging Subset of DDSM)**, which provides mammograms labeled with tumor characteristics.

---

## Project Structure

```
tumor_clasification/
│
├── dataset/
│   ├── folder_organizator.py       # Organizes and preprocesses DICOM images
│   └── image_procesor.py           # Loads and normalizes images for training/validation
│
├── models/
│   ├── augmentation/               # Models using data augmentation techniques
│   ├── batch_normalization/        # Models with batch normalization
│   ├── dropout/                    # Models with dropout regularization
│   ├── early_stopping/             # Models trained with early stopping
│   ├── regularizationl1/           # L1 regularization
│   ├── regularizationl2/           # L2 regularization
│   ├── regularizationl1l2/         # Combined L1+L2 regularization
│   ├── initial_model.py            # Base multimodal model
│   └── utils_models.py             # Utility functions (visualization, metrics, etc.)
```

---

## Data Preprocessing

The `folder_organizator.py` script organizes DICOM images into folders by type (full image, ROI, cropped) and converts them to PNG format. Images are resized while preserving aspect ratio and padded with black borders.

```bash
python dataset/folder_organizator.py
```

Then, the `image_procesor.py` script loads and prepares the images for training. It:

* Converts images to grayscale.
* Adjusts dimensions and normalizes pixel values.
* Returns tensors ready for model input.

---

## Multimodal Model

The `initial_model.py` file defines the base model, which takes two inputs:

* **Full mammogram** (800x1350)
* **Cropped region** (550x550)

Each input follows a parallel convolutional path. The outputs are concatenated to produce the final prediction. The model is trained for 10 epochs using `categorical_crossentropy` and the `Adam` optimizer.

---

## Optimization Techniques

Several versions of the base model are included, each using different optimization strategies:

| Folder                 | Applied Technique                         |
| ---------------------- | ----------------------------------------- |
| `augmentation/`        | Data augmentation (rotation, flip, etc.)  |
| `batch_normalization/` | Batch normalization                       |
| `dropout/`             | Dropout in dense and convolutional layers |
| `early_stopping/`      | Training with early stopping              |
| `regularizationl1/`    | L1 regularization                         |
| `regularizationl2/`    | L2 regularization                         |
| `regularizationl1l2/`  | Combined L1 and L2 regularization         |

---

## Results and Metrics

During training, **loss** and **accuracy** for both training and validation sets are plotted using a utility function from `utils_models.py`.

---

## Execution

1. Preprocess DICOM images:

   ```bash
   python dataset/folder_organizator.py
   ```

2. Load and normalize images:

   ```bash
   python dataset/image_procesor.py
   ```

3. Train the base model:

   ```bash
   python models/initial_model.py
   ```

4. Test model variants (e.g., regularization, dropout) by running the corresponding script.

---

## Requirements

* Python 3.8+
* TensorFlow 2.x
* Pillow
* NumPy
* OpenCV
* pydicom

Installation:

```bash
pip install tensorflow pillow numpy opencv-python pydicom
```
