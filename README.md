# 💵 AI Dollar Bill Value Detector

An automated image classification system designed to detect and identify the denomination of dollar bills ($1, $5, $10, $20) using a Deep Learning approach.

## 🚀 Project Overview
This project addresses the challenge of currency recognition using Computer Vision. After comparing various architectures, a custom Convolutional Neural Network (CNN) was implemented to achieve optimal performance on a specialized dataset of 130 images.

### Key Performance Metrics
* **Test Accuracy:** 93.55%
* **Architecture:** Optimized Simple CNN (32-64 Layer Design)
* **Training Time:** 5 Epochs
* **Data Integrity:** Strictly partitioned Train/Test sets with zero data leakage.
---
## 📁 Repository Structure
```text
├── data/                   # Dataset directory
│   ├── currency_data/      # Training images (1, 5, 10, 20)
│   └── test_data/          # Validation images (Mirrored structure)
├── models/                 # Pre-trained model storage
│   └── dollar_detector.keras
├── scripts/                # Python execution scripts
│   ├── train.py            # Model training & splitting logic
│   └── predict.py          # Single image inference script
├── README.md               # Project documentation
└── requirements.txt        # Dependency list
```
---
## 🛠️ Installation & Setup
Clone the Repository:

```
git clone [https://github.com/Muntazir-43/Computer-VIsion-Quiz-02.git](https://github.com/Muntazir-43/Computer-VIsion-Quiz-02.git)
cd dollar-bill-detector

```
## Install Dependencies:

```
pip install -r requirements.txt
Train the Model (Optional):
```

## python scripts/train.py
🔍 How to Verify the Model
To test the model with a custom image, use the provided prediction script:

```
python scripts/predict.py

```

The script will output the Predicted Value and the Confidence Score (%) based on the trained weights.

---

## 🧠 Methodology
The model utilizes a sequence of Conv2D and MaxPooling2D layers to extract spatial features from the currency images.

Normalization: Pixel values are rescaled to the [0, 1] range.

Classification: A Dense softmax layer outputs probabilities for 4 classes (1, 5, 10, 20).

---

## 👤 Author Information

**Muntazir Mehdi**

**2022-SE-37**
---

## 📜 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
