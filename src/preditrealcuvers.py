import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix

# --------------- CONFIG ----------------
ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"
RESULT_DIR = ROOT_DIR / "result"
PLOTS_DIR = RESULT_DIR / "plots_real"

CSV_PATH = DATA_DIR / "lightcurves_train_clean.csv"
MODEL_PATH = DATA_DIR / "lightcurve_cnn_advanced_final.keras"
PLOT_DIR = PLOTS_DIR
OUTPUT_CSV = RESULT_DIR / "predictions_real.csv"

SHOW_PLOTS = False
MAX_LEN = 512
# ---------------------------------------

os.makedirs(PLOT_DIR, exist_ok=True)

print("🧠 Caricamento modello...")
model = tf.keras.models.load_model(MODEL_PATH)
mapping_inv = {0: "PC", 1: "FP", 2: "EB"}

# Caricamento CSV
print(f"📄 Caricamento curve da {CSV_PATH}...")
df_raw = pd.read_csv(CSV_PATH)

# Raggruppa per curva_id (una riga per punto -> 512 per curva)
print("🔄 Preparazione curve...")
curves = []
labels = []
ids = []

for curve_id, group in df_raw.groupby("curve_id"):
    flux = group["flux"].values
    if len(flux) != MAX_LEN:
        print(f"⛔ Curve {curve_id} ha lunghezza {len(flux)} (attesa {MAX_LEN}), salto.")
        continue
    X = flux.reshape(-1, 1).astype("float32")
    label = group["label"].iloc[0]
    curves.append(X)
    labels.append(label)
    ids.append(curve_id)

print(f"✅ Trovate {len(curves)} curve valide")

# Inference
print("🧪 Inference in corso...")
X_all = np.array(curves)
preds = model.predict(X_all, verbose=0)
pred_classes = np.argmax(preds, axis=1)
pred_labels = [mapping_inv[i] for i in pred_classes]

# Report
df_results = pd.DataFrame({
    "curve_id": ids,
    "true_label": labels,
    "predicted_label": pred_labels,
    "prob_PC": preds[:, 0],
    "prob_FP": preds[:, 1],
    "prob_EB": preds[:, 2]
})

df_results.to_csv(OUTPUT_CSV, index=False)
print(f"\n📁 Risultati salvati in '{OUTPUT_CSV}'")
#
## Plot
print("📈 Salvataggio grafici...")
for i, flux in enumerate(curves):
    print(f"Salvando plot {i+1}/{len(curves)} id={ids[i]}")
    plt.figure(figsize=(8, 3))
    plt.plot(flux)
    title = f"{ids[i]} [{labels[i]}] → {pred_labels[i]}"
    plt.title(title)
    plt.xlabel("Fasi (normalizzate)")
    plt.ylabel("Flux (z-score)")
    plt.grid()
    plt.tight_layout()
    filename = f"{PLOT_DIR}/{ids[i]}_{pred_labels[i]}.png"
    # Sanifica il filename da caratteri strani
    filename = filename.replace(" ", "_").replace("/", "_")
    plt.savefig(filename)
    if SHOW_PLOTS:
        plt.show()
    plt.close()
#
#
## Metrics
print("\n📊 Classification Report:")
print(classification_report(labels, pred_labels, digits=4))

print("\n📉 Confusion Matrix:")
cm = confusion_matrix(labels, pred_labels, labels=["PC", "FP", "EB"])
print(pd.DataFrame(cm, index=["true_PC", "true_FP", "true_EB"], columns=["pred_PC", "pred_FP", "pred_EB"]))
