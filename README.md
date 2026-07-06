# 🫀 ECG Arrhythmia Classification — CNN & LSTM

> An AI-powered system for detecting and classifying cardiac arrhythmias from ECG signals using Deep Learning.

---

## 📌 Project Overview

This project builds an intelligent classification system for ECG (Electrocardiogram) heartbeat signals using two deep learning architectures — **CNN** and **LSTM**. The system analyzes 187-timestep ECG signals and classifies them into 5 cardiac categories, enabling early detection of life-threatening arrhythmias.

---

## 🎯 Motivation

- Over **17 million deaths** per year are caused by cardiovascular disease worldwide
- Manual ECG interpretation requires a **specialized cardiologist** — not always available
- AI-based systems can analyze thousands of ECGs **in seconds**, 24/7, with no fatigue
- Early detection of arrhythmias like **Ventricular Contractions** can be life-saving

---

## 📊 Dataset

**ECG Heartbeat Categorization Dataset** — [Kaggle (MIT-BIH Arrhythmia)](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)

| File | Samples | Shape |
|------|---------|-------|
| `mitbih_train.csv` | 87,554 | (87554, 188) |
| `mitbih_test.csv` | 21,892 | (21892, 188) |

Each row = one heartbeat (187 signal values + 1 label)

### Classes

| Label | Class | Description | Risk |
|-------|-------|-------------|------|
| 0 | N — Normal | Normal sinus beat | ✅ None |
| 1 | S — Supraventricular | Premature atrial beat | ⚠️ Moderate |
| 2 | V — Ventricular | Premature ventricular contraction | 🔴 High |
| 3 | F — Fusion | Fusion of ventricular and normal beat | 🔴 High |
| 4 | Q — Unknown | Unclassifiable / paced beat | ❓ Unknown |

---

## ⚙️ Project Pipeline

```
Raw ECG Signal (187 timesteps)
        ↓
  Data Cleaning
  (duplicates, missing values, inf)
        ↓
  Normalization (MinMaxScaler)
        ↓
  Class Balancing (SMOTE + Undersampling)
  Normal: 63,000 → 15,000
  Minority classes → 6,000 each
        ↓
  Model Training
  ├── CNN (1D Convolutional)
  └── LSTM (Bidirectional)
        ↓
  Threshold Tuning
  (per-class probability thresholds)
        ↓
  Evaluation & Comparison
```

---

## 🧠 Models

### CNN — 1D Convolutional Neural Network
- 3 Conv1D blocks (64 → 128 → 256 filters)
- BatchNormalization + Dropout (anti-overfitting)
- GlobalAveragePooling + Dense head
- **Focal Loss** (γ=2.0) to handle class imbalance

### LSTM — Bidirectional Long Short-Term Memory
- 2 Bidirectional LSTM layers (64 + 32 units)
- Recurrent Dropout
- Dense classification head
- Categorical Cross-Entropy loss

---

## 📈 Results

### CNN — Before vs After Optimization

| Class | Precision (Before) | Precision (After) |
|-------|-------------------|-------------------|
| Normal (N) | 0.9958 | 0.9948 ✅ |
| Supraventricular (S) | 0.5397 | 0.6640 🟡 |
| Ventricular (V) | 0.9592 | 0.9539 ✅ |
| Fusion (F) | 0.3680 | 0.5814 🟡 |
| Unknown (Q) | 0.9702 | 0.9834 ✅ |
| **Macro F1** | 0.8222 | **0.8796** |

### CNN vs LSTM Comparison

| Metric | CNN | LSTM |
|--------|-----|------|
| Accuracy | **0.9605** | 0.8344 |
| Weighted F1 | **0.9656** | 0.8646 |
| Macro F1 | **0.8796** | 0.6209 |

> **CNN outperforms LSTM** on this dataset due to the strong local pattern structure of ECG signals.

---

## 🛠️ Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| **Severe class imbalance** (Normal = 72%) | SMOTE oversampling + RandomUnderSampler for Normal |
| **Low precision on minority classes** | Focal Loss with per-class alpha weights |
| **Model bias toward Normal class** | Strong class weights (Fusion=25x, Supra=15x) |
| **Default argmax poor for rare classes** | Per-class threshold tuning (sweep 256 combinations) |
| **Overfitting risk** | BatchNorm + Dropout + L2 + EarlyStopping + ReduceLR |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/ECG-Arrhythmia-Classification.git
cd ECG-Arrhythmia-Classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download dataset
```bash
kaggle datasets download -d shayanfazeli/heartbeat
unzip heartbeat.zip
```

### 4. Run the notebook
Open `ECG_CNN_LSTM.ipynb` in Google Colab or Jupyter and run all cells.

---

## 📁 Repository Structure

```
ECG-Arrhythmia-Classification/
│
├── ECG_CNN_LSTM.ipynb          # Main notebook (all milestones)
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── presentation/
    └── ECG_Presentation.pptx   # Project presentation
```

---

## 🧰 Tech Stack

| Tool | Purpose |
|------|---------|
| TensorFlow / Keras | Deep learning models |
| scikit-learn | Preprocessing & metrics |
| imbalanced-learn | SMOTE oversampling |
| NumPy / Pandas | Data manipulation |
| Matplotlib / Seaborn | Visualization |
| Google Colab | Training environment (T4 GPU) |

---

## 📚 Milestones

| Milestone | Description | Status |
|-----------|-------------|--------|
| **M1** | Data Collection & EDA | ✅ Done |
| **M2** | Preprocessing & Balancing | ✅ Done |
| **M3** | CNN & LSTM Model Development | ✅ Done |
| **M4** | Evaluation & Optimization | ✅ Done |

---

## 📄 References

- [MIT-BIH Arrhythmia Dataset](https://www.physionet.org/content/mitdb/1.0.0/)
- [ECG Heartbeat Categorization — Kaggle](https://www.kaggle.com/datasets/shayanfazeli/heartbeat)
- Lin et al. (2017) — Focal Loss for Dense Object Detection
- Kachuee et al. (2018) — ECG Heartbeat Classification: A Deep Transferable Representation

---

*Graduation Project — Smart Medical System for ECG Analysis*
