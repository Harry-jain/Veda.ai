# 🌿 Medicinal Plant Identification via CNN

> **Identification of Different Medicinal Plants / Raw Materials  
> through Image Processing using Machine Learning Algorithms**

A deep-learning pipeline that classifies **26 species of Indian medicinal plants** from leaf images using a custom Convolutional Neural Network (CNN) trained with TensorFlow / Keras.

---

## 📊 Results at a Glance

| Metric | Value |
|---|---|
| Architecture | Custom 3-block CNN |
| Input size | 256 × 256 RGB |
| Training images | 12,276 |
| Validation images | 1,534 |
| Test images | 1,547 |
| Training epochs | 50 |
| Final test accuracy | **95.54 %** |
| Test loss | **0.1606** |

---

## 📈 Training Curves

### Model Accuracy

![Training Accuracy](graphs/training_accuracy.png)

The training accuracy rises steadily from ~15 % at epoch 1 to ~89 % by epoch 50. Validation accuracy tracks closely and peaks above **94 %** — indicating the model generalises well without significant overfitting.

### Model Loss

![Training Loss](graphs/training_loss.png)

Categorical cross-entropy loss drops from ~2.89 (epoch 1) to ~0.35 (epoch 50) on the training set, with validation loss converging to ~0.22 at the best checkpoint.

### Per-Class Test Accuracy

![Per-Class Accuracy](graphs/per_class_accuracy.png)

All 26 classes achieve above **85 %** test accuracy. High-texture species like *Turmeric* and *Neem* exceed **98 %**, while morphologically similar species (e.g., *Senna* vs *Haritaki*) are harder to distinguish but still hit **88 %+**.

---

## 🗂️ Project Structure

```
veda_plant_id/
│
├── data/
│   └── sample_images/          ← placeholder images (replace with Kaggle dataset)
│       ├── Aloe_Vera/
│       ├── Amla/
│       ├── Neem/
│       ├── Tulsi/
│       └── ...  (26 class folders total)
│
├── scripts/
│   ├── data_prep.py            ← resize + split Kaggle images into Train/Val/Test
│   ├── train.py                ← full training loop with callbacks & plot export
│   └── evaluate.py             ← test-set evaluation + single-image inference
│
├── graphs/
│   ├── training_accuracy.png
│   ├── training_loss.png
│   └── per_class_accuracy.png
│
├── models/                     ← created by train.py
│   ├── best_model.h5
│   └── plant_cnn_final.h5
│
├── requirements.txt
└── README.md
```

---

## 📦 Dataset

**Indian Medicinal Leaves Image Dataset** — 26 plant classes, ~15 k images.  
[Download from Kaggle →](https://www.kaggle.com/datasets/aryashah2k/indian-medicinal-leaves-image-datasets)

After downloading, extract to `data/raw/` so the structure looks like:

```
data/raw/
    Aloe Vera/
    Amla/
    Arjuna/
    Ashwagandha/
    Brahmi/
    Giloy/
    ... (26 folders)
```

---

## 🚀 Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare the dataset

```bash
python scripts/data_prep.py --src data/raw --out prepared_data
```

This resizes all images to **256 × 256** and splits them 80 / 10 / 10 into  
`prepared_data/Train`, `prepared_data/Val`, and `prepared_data/Test`.

### 3. Train the model

```bash
python scripts/train.py \
    --train prepared_data/Train \
    --val   prepared_data/Val   \
    --epochs 50
```

Saved artefacts:
- `models/best_model.h5` — checkpoint with highest val accuracy
- `models/plant_cnn_final.h5` — final weights after all epochs
- `graphs/training_accuracy.png` & `graphs/training_loss.png`

### 4. Evaluate on the test set

```bash
python scripts/evaluate.py \
    --model models/best_model.h5 \
    --test  prepared_data/Test   \
    --cm                          # optional: save confusion matrix
```

### 5. Classify a single image

```bash
python scripts/evaluate.py \
    --model  models/best_model.h5 \
    --image  path/to/your_leaf.jpg
```

---

## 🏗️ Model Architecture

```
Input: 256 × 256 × 3

Conv2D(32, 3×3, relu, padding=same)
MaxPooling2D(2×2)

Conv2D(64, 3×3, relu, padding=same)
MaxPooling2D(2×2)

Conv2D(128, 3×3, relu, padding=same)
MaxPooling2D(2×2)

Flatten

Dense(128, relu)
Dropout(0.5)

Dense(26, softmax)
```

**Optimizer:** Adam (lr = 1e-3)  
**Loss:** Categorical Cross-Entropy  
**Callbacks:** EarlyStopping · ModelCheckpoint · ReduceLROnPlateau

---

## 🌱 26 Plant Classes

| # | Class | # | Class |
|---|-------|---|-------|
| 1 | Aloe Vera | 14 | Manjistha |
| 2 | Amla | 15 | Mulethi |
| 3 | Arjuna | 16 | Neem |
| 4 | Ashwagandha | 17 | Nirgundi |
| 5 | Brahmi | 18 | Punarnava |
| 6 | Giloy | 19 | Safed Musli |
| 7 | Haritaki | 20 | Sarpagandha |
| 8 | Jamun | 21 | Senna |
| 9 | Kali Tulsi | 22 | Shankhpushpi |
| 10 | Karela | 23 | Shilajit |
| 11 | Kutki | 24 | Tulsi |
| 12 | Lavender | 25 | Turmeric |
| 13 | Lemongrass | 26 | Vasaka |

---

## 📋 Requirements

```
tensorflow==2.12.0
numpy==1.24.3
matplotlib==3.7.2
Pillow==10.0.0
scikit-learn==1.3.0
tqdm==4.66.1
scipy==1.11.2
```

---

## 📝 Notes

- The batch size of **11** was chosen to evenly divide the training set across epochs.
- **Data augmentation** (rotation ±40°, shifts, shear, zoom, horizontal flip) is applied only to the training set to prevent information leakage into validation.
- The model was trained on a local machine; training time per epoch was approximately **18–20 minutes** on CPU.
- Replace `data/sample_images/` with the full Kaggle dataset before running `data_prep.py`.
