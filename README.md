# 🔭 Pulsar Detection — Hybrid CNN-LSTM-Attention Model

> Detecting pulsar stars from the HTRU2 dataset using a hybrid deep learning architecture with GAN-based synthetic data generation and SMOTE to handle severe class imbalance.

---

## 📌 Overview

Pulsars are rapidly rotating neutron stars that emit beams of electromagnetic radiation detectable from Earth. Identifying genuine pulsar candidates from large-scale radio survey data is challenging — most detections are caused by radio frequency interference (RFI) or noise.

This project tackles two key challenges:

- **Class imbalance** — The HTRU2 dataset has a severe 91:9 ratio (non-pulsar:pulsar). A naive model would simply predict "not a pulsar" and achieve 91% accuracy while missing every real pulsar.
- **Detection robustness** — Using a hybrid CNN-LSTM-Attention architecture to capture both spatial patterns and temporal dependencies in pulsar signal features.

---

## 🏗️ Architecture

```
HTRU2 Dataset (17,898 samples)
        │
        ▼
┌───────────────────────────────────────────┐
│         Class Imbalance Handling          │
│                                           │
│   GAN → Synthetic minority samples        │
│   SMOTE → Further oversampling            │
│                                           │
│   Before: 16,259 non-pulsar / 1,639 pulsar│
│   After:  Balanced training distribution  │
└───────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────┐
│       Hybrid Deep Learning Model        │
│                                         │
│   CNN Layer → Local feature extraction  │
│   LSTM Layer → Temporal dependencies    │
│   Attention Layer → Weighted focus      │
│   Dense + Sigmoid → Binary output       │
└─────────────────────────────────────────┘
        │
        ▼
   Pulsar / Non-Pulsar
```

---

## 📊 Results

| Metric | Score |
|---|---|
| Overall Accuracy | **90%+** |
| Precision (Pulsar class) | 86% |
| Recall (Pulsar class) | 95% |
| Class Imbalance Ratio | 91:9 → Balanced (post-GAN + SMOTE) |

> The focus metric is **minority-class recall** — in pulsar detection, missing a real pulsar (false negative) is far more costly than a false alarm.

---

## 📁 Repository Structure

```
Pulsar-Detection/
├── PULSAR.ipynb                    # Main model: CNN-LSTM-Attention pipeline
├── Synthetic_dataset_using_GAN.ipynb  # GAN training for synthetic minority samples
└── README.md
```

---

## 🗃️ Dataset

**HTRU2** — High Time Resolution Universe Survey (South)

- 17,898 samples, 8 features per candidate
- 1,639 positive (pulsar), 16,259 negative (non-pulsar)
- Features: mean/standard deviation/excess kurtosis/skewness of integrated pulse profile and DM-SNR curve

📥 [Download from UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/HTRU2)

---

## 🧪 Methodology

### 1. Data Preprocessing
- Feature scaling using StandardScaler
- Exploratory analysis: class distribution, correlation heatmaps, feature distributions

### 2. Class Imbalance Handling
- **GAN (Generative Adversarial Network)** — Trained a generator to produce realistic synthetic pulsar samples mimicking the statistical properties of real minority-class instances
- **SMOTE (Synthetic Minority Over-sampling Technique)** — Applied after GAN augmentation for further balancing

### 3. Model Architecture
- **CNN layers** — Extract local patterns from the 8-feature input
- **LSTM layers** — Learn sequential/temporal dependencies
- **Attention mechanism** — Enables the model to weight features by importance
- **Output** — Binary classification (pulsar / non-pulsar)

### 4. Evaluation
- Confusion matrix, classification report (precision, recall, F1)
- Emphasis on minority-class recall to minimize missed pulsars

---

## ⚙️ How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the notebooks

```bash
# Step 1 — Generate synthetic data
jupyter notebook Synthetic_dataset_using_GAN.ipynb

# Step 2 — Train and evaluate the main model
jupyter notebook PULSAR.ipynb
```

---

## 📦 Requirements

```
tensorflow>=2.10
numpy
pandas
scikit-learn
imbalanced-learn
matplotlib
seaborn
jupyter
```

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat-square)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-yellow?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square)

**Models:** CNN · LSTM · Attention · GAN  
**Libraries:** TensorFlow, Scikit-learn, imbalanced-learn (SMOTE), NumPy, Pandas, Matplotlib, Seaborn  
**Dataset:** HTRU2 (UCI ML Repository)

---
