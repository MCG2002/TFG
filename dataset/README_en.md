# Dataset

This project uses the **CBIS-DDSM: Curated Breast Imaging Subset of DDSM**, developed by Sawyer-Lee et al. (2016). It is a curated and updated version of the classic **DDSM** (Digital Database for Screening Mammography), with improvements in annotation quality, accessibility, and segmentation.

The dataset includes **10,237 mammograms** from **1,566 patients** and categorizes the images into three types:

* **Full mammograms**
* **Cropped images** focusing on the suspicious region
* **Segmented ROI masks** showing the tumor location

| Image Type      | Number of Samples |
| --------------- | ----------------- |
| Full Mammograms | 2,857             |
| Cropped Images  | 3,567             |
| ROI Masks       | 3,247             |
| Invalid Samples | 566               |

---

## Additional Files

The dataset includes two CSV files:

* **Calcifications** (not used in this project)
* **Mass lesions** (used in this project)

These files contain detailed metadata on:

* **Pathology**: Breast masses are classified into three main categories based on pathology:

  * **Malignant**: Cancerous growth with uncontrolled cell division and invasive characteristics.
  * **Benign**: Non-cancerous masses, generally slow-growing with well-defined borders and no tissue invasion.
  * **Benign without further evaluation**: Benign lesions considered stable and requiring no further follow-up.

* **BI-RADS Category**: A standardized reporting system used by radiologists to describe mammographic findings, known as BI-RADS (Breast Imaging Reporting and Data System). Categories range from 0 to 6:

  * **0**: Incomplete — additional imaging or prior comparisons required.
  * **1**: Negative result.
  * **2**: Benign finding.
  * **3**: Probably benign — short-term follow-up may be needed.
  * **4**: Suspicious abnormality — biopsy should be considered.
  * **5**: Highly suggestive of malignancy.
  * **6**: Known biopsy-proven malignancy.

  These categories aid communication and guide clinical decision-making.

* **Mass Shape**: Includes categories such as:

  * **Spiculated**: Irregular, spike-like borders.
  * **Circumscribed**: Well-defined, rounded borders.
  * **Ill-defined**: Poorly visible or fuzzy borders.
  * **Obscured**: Hidden by surrounding tissue or image quality.
  * **Microlobulated**: Small lobes visible along the edges of the mass.

* **Margins**: Categories include:

  * **Circumscribed**, **spiculated**, **ill-defined**, **obscured**, **microlobulated**, and combinations (e.g., circumscribed/ill-defined).

* **Breast Density**: Four levels of mammographic density are identified and numbered:

  * (1) Fatty
  * (2) Scattered fibroglandular
  * (3) Heterogeneously dense
  * (4) Extremely dense

  Density is classified as **low** for values 1–2 and **high** for values 3–4.

---

## Derived Classifications

Based on the CSV metadata, an **extended classification** of tumors has been defined. This offers more precise labels than the original categorization, factoring in both pathology and estimated clinical stage.

A total of **11 final classes** are identified:

* Stable Benign
* Benign
* Dangerous Benign
* Malignant Stage 0-1
* Malignant Stage 1-2
* Malignant Stage 2
* Malignant Stage 2-3
* Malignant Stage 3
* Malignant Stage 3-4
* Malignant Stage 4
* Incomplete Diagnosis (excluded from training)

### Class Distribution

| Class                | Percentage |
| -------------------- | ---------- |
| Stable Benign        | 7.89%      |
| Benign               | 20.94%     |
| Dangerous Benign     | 16.08%     |
| Malignant Stage 3-4  | 23.07%     |
| Malignant Stage 4    | 22.23%     |
| Incomplete Diagnosis | 9.79%      |

The remaining classes are not represented in the training set.

---

## Preprocessing

The dataset is preprocessed to ensure consistency, reduce computational load, and optimize model performance.

### Applied Techniques

* **Class-Based Organization**: Images are grouped into three main folders — `full mammogram images`, `cropped images`, and `ROI mask images` — each further subdivided into the 11 defined classes.

* **Resizing with Padding**:

  * Full mammograms and ROI masks: `800x1350 px`
  * Cropped images: `550x550 px`
  * The aspect ratio is preserved; black padding is added as needed.

* **Normalization**: Pixel values are scaled to the range `[0, 1]`.

* **Format Conversion**: All images are converted from DICOM to PNG for easier handling and compatibility with deep learning frameworks.
