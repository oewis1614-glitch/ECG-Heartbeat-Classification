🫀 ECG Arrhythmia Classification — CNN & LSTM


An AI-powered system for detecting and classifying cardiac arrhythmias from ECG signals using Deep Learning.




📌 Project Overview

This project builds an intelligent classification system for ECG (Electrocardiogram) heartbeat signals using two deep learning architectures — CNN and LSTM. The system analyzes 187-timestep ECG signals and classifies them into 5 cardiac categories, enabling early detection of life-threatening arrhythmias.


🎯 Motivation


Over 17 million deaths per year are caused by cardiovascular disease worldwide
Manual ECG interpretation requires a specialized cardiologist — not always available
AI-based systems can analyze thousands of ECGs in seconds, 24/7, with no fatigue
Early detection of arrhythmias like Ventricular Contractions can be life-saving



📊 Dataset

ECG Heartbeat Categorization Dataset — Kaggle (MIT-BIH Arrhythmia)

FileSamplesShapemitbih_train.csv87,554(87554, 188)mitbih_test.csv21,892(21892, 188)

Each row = one heartbeat (187 signal values + 1 label)

Classes

LabelClassDescriptionRisk0N — NormalNormal sinus beat✅ None1S — SupraventricularPremature atrial beat⚠️ Moderate2V — VentricularPremature ventricular contraction🔴 High3F — FusionFusion of ventricular and normal beat🔴 High4Q — UnknownUnclassifiable / paced beat❓ Unknown


⚙️ Project Pipeline

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


🧠 Models

CNN — 1D Convolutional Neural Network


3 Conv1D blocks (64 → 128 → 256 filters)
BatchNormalization + Dropout (anti-overfitting)
GlobalAveragePooling + Dense head
Focal Loss (γ=2.0) to handle class imbalance


LSTM — Bidirectional Long Short-Term Memory


2 Bidirectional LSTM layers (64 + 32 units)
Recurrent Dropout
Dense classification head
Categorical Cross-Entropy loss



📈 Results

CNN — Before vs After Optimization

ClassPrecision (Before)Precision (After)Normal (N)0.99580.9948 ✅Supraventricular (S)0.53970.6640 🟡Ventricular (V)0.95920.9539 ✅Fusion (F)0.36800.5814 🟡Unknown (Q)0.97020.9834 ✅Macro F10.82220.8796

CNN vs LSTM Comparison

MetricCNNLSTMAccuracy0.96050.8344Weighted F10.96560.8646Macro F10.87960.6209


CNN outperforms LSTM on this dataset due to the strong local pattern structure of ECG signals.




🛠️ Challenges & Solutions

ChallengeSolutionSevere class imbalance (Normal = 72%)SMOTE oversampling + RandomUnderSampler for NormalLow precision on minority classesFocal Loss with per-class alpha weightsModel bias toward Normal classStrong class weights (Fusion=25x, Supra=15x)Default argmax poor for rare classesPer-class threshold tuning (sweep 256 combinations)Overfitting riskBatchNorm + Dropout + L2 + EarlyStopping + ReduceLR
