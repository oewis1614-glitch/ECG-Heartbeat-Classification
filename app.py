import gradio as gr
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import os
import spaces

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

CLASS_MAPPING = {
    0: "Normal Beat (N)",
    1: "Supraventricular Premature Beat (S)",
    2: "Premature Ventricular Contraction (V)",
    3: "Fusion of Ventricular and Normal Beat (F)",
    4: "Unclassifiable Beat (Q)"
}
CLASS_RISK = {
    0: "✅ Normal — No action required",
    1: "⚠️ Moderate — Monitor and consult physician",
    2: "🔴 High Risk — Immediate medical attention",
    3: "🔴 High Risk — Immediate medical attention",
    4: "❓ Unknown — Further evaluation needed"
}
COLORS = ['#2ecc71','#3498db','#e74c3c','#f39c12','#9b59b6']

# ── Load model lazily inside GPU function ─────────────────────────────────────
model = None

@spaces.GPU
def predict_ecg(text_input):
    global model
    if model is None:
        import tensorflow as tf
        from tensorflow.keras import layers, models, regularizers
        m = models.Sequential([
            layers.Input(shape=(187, 1)),
            layers.Conv1D(64, 7, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(), layers.MaxPooling1D(2), layers.Dropout(0.2),
            layers.Conv1D(128, 5, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(), layers.MaxPooling1D(2), layers.Dropout(0.25),
            layers.Conv1D(256, 3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(), layers.MaxPooling1D(2), layers.Dropout(0.3),
            layers.Conv1D(256, 3, padding='same', activation='relu',
                          kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(), layers.GlobalAveragePooling1D(), layers.Dropout(0.3),
            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
            layers.BatchNormalization(), layers.Dropout(0.4),
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dropout(0.3),
            layers.Dense(5, activation='softmax')
        ], name="CNN_ECG")
        m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        m.load_weights("cnn_weights.weights.h5")
        model = m

    try:
        cleaned = text_input.replace(",", " ").replace("\n", " ").replace("\t", " ")
        parsed  = [float(x) for x in cleaned.split() if x.strip()]
        count   = len(parsed)

        if count not in (187, 188):
            return None, f"❌ Expected 187 or 188 values, got {count}.", "", ""

        features   = np.array(parsed[:187], dtype=np.float32)
        true_label = int(parsed[187]) if count == 188 else None

        scaler        = MinMaxScaler()
        signal_scaled = scaler.fit_transform(features.reshape(-1, 1)).reshape(1, 187, 1)
        probs         = model.predict(signal_scaled, verbose=0)[0]
        pred_class    = int(np.argmax(probs))
        confidence    = float(np.max(probs)) * 100

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        fig.patch.set_facecolor('#0e1117')

        ax1 = axes[0]
        ax1.set_facecolor('#1a1a2e')
        ax1.plot(features, color='#00ff88', linewidth=1.8)
        ax1.fill_between(range(187), features, alpha=0.15, color='#00ff88')
        r_idx = int(np.argmax(features))
        ax1.axvline(r_idx, color='red', linestyle='--', alpha=0.7)
        ax1.scatter(r_idx, features[r_idx], color='red', s=80, zorder=5)
        ax1.set_title("ECG Signal", color='white', fontsize=13, fontweight='bold')
        ax1.set_xlabel("Time Steps", color='#aaa')
        ax1.set_ylabel("Amplitude", color='#aaa')
        ax1.tick_params(colors='#aaa')
        ax1.grid(True, alpha=0.2, color='#444')
        for sp in ax1.spines.values(): sp.set_edgecolor('#333')

        ax2 = axes[1]
        ax2.set_facecolor('#1a1a2e')
        bars = ax2.barh([CLASS_MAPPING[i] for i in range(5)],
                        probs * 100, color=COLORS, alpha=0.85)
        for bar, val in zip(bars, probs):
            ax2.text(val*100+0.5, bar.get_y()+bar.get_height()/2,
                     f"{val*100:.1f}%", va='center', color='white', fontsize=9)
        ax2.set_xlabel("Probability (%)", color='#aaa')
        ax2.set_title("Class Probabilities", color='white', fontsize=13, fontweight='bold')
        ax2.set_xlim(0, 115)
        ax2.tick_params(colors='#aaa')
        ax2.grid(True, alpha=0.2, axis='x', color='#444')
        for sp in ax2.spines.values(): sp.set_edgecolor('#333')

        plt.tight_layout()

        result   = f"**🤖 Predicted:** {CLASS_MAPPING[pred_class]}\n\n**📊 Confidence:** {confidence:.2f}%\n\n**⚕️ Risk:** {CLASS_RISK[pred_class]}"
        true_out = ""
        if true_label is not None:
            match    = "✅ Correct!" if pred_class == true_label else "❌ Incorrect"
            true_out = f"**🏷️ True Label:** {CLASS_MAPPING.get(true_label,'Unknown')} — {match}"
        stats = f"**Signal Stats:** R-peak={features.max():.4f} @ t={r_idx} | Min={features.min():.4f} | Mean={features.mean():.4f} | Std={features.std():.4f}"

        return fig, result, true_out, stats

    except Exception as e:
        return None, f"❌ Error: {str(e)}", "", ""

# ── UI ────────────────────────────────────────────────────────────────────────
with gr.Blocks(title="ECG Classification") as demo:
    gr.Markdown("""
# 🫀 ECG Arrhythmia Classification
**1D CNN | MIT-BIH Dataset | Accuracy: 97.25% | Macro F1: 0.8796**
---
""")
    with gr.Row():
        with gr.Column(scale=1):
            text_input  = gr.Textbox(
                label="Paste 187 or 188 ECG values",
                lines=8,
                placeholder="0.5, 0.8, 0.3, 1.0 ..."
            )
            predict_btn = gr.Button("▶ Run Prediction", variant="primary")
            gr.Markdown("""
| Label | Class | Risk |
|-------|-------|------|
| 0 | Normal (N) | ✅ |
| 1 | Supraventricular (S) | ⚠️ |
| 2 | Ventricular (V) | 🔴 |
| 3 | Fusion (F) | 🔴 |
| 4 | Unknown (Q) | ❓ |
""")
        with gr.Column(scale=2):
            plot_out   = gr.Plot(label="ECG Signal & Probabilities")
            result_out = gr.Markdown()
            true_out   = gr.Markdown()
            stats_out  = gr.Markdown()

    predict_btn.click(
        predict_ecg,
        inputs=[text_input],
        outputs=[plot_out, result_out, true_out, stats_out]
    )
    gr.Markdown("*DEPI Final Project 2025*")

demo.launch(ssr_mode=False)