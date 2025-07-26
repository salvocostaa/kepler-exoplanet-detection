#!/usr/bin/env python
"""
train_lightcurve_cnn_advanced.py
CNN + Transformer per classificazione light curve Kepler
"""

import numpy as np
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models, callbacks, regularizers
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------- 1. Caricamento dati --------------------
ROOT_DIR = Path(__file__).parent.parent.resolve()

DATA_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = DATA_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# -------------------- 1. Caricamento dati --------------------
X_train = np.load(DATA_DIR / "X_train.npy")
y_train = np.load(DATA_DIR / "y_train.npy")
X_test = np.load(DATA_DIR / "X_test.npy")
y_test = np.load(DATA_DIR / "y_test.npy")
print("X_train:", X_train.shape, " y_train:", y_train.shape)
print("X_test :", X_test.shape,  " y_test :", y_test.shape)

if X_train.ndim == 2:
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

if y_train.ndim == 1:
    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
else:
    num_classes = y_train.shape[1]

# Class weights per dataset sbilanciato
y_int = np.argmax(y_train, axis=1)
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y_int),
                                     y=y_int)
class_weights = dict(enumerate(class_weights))
print("✓ Class weights:", class_weights)

# -------------------- 2. Modello CNN + Transformer --------------------
def build_model(input_shape, n_classes):
    inp = layers.Input(shape=input_shape)

    # --- CNN stack ---
    x = layers.Conv1D(64, 7, padding='same', activation='relu')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(2)(x)

    x = layers.Conv1D(256, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)

    # --- Transformer block ---
    attn = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)

    # --- Output layers ---
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)

    out = layers.Dense(n_classes, activation='softmax')(x)

    model = models.Model(inputs=inp, outputs=out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )
    return model

model = build_model(X_train.shape[1:], num_classes)
model.summary()

# -------------------- 3. Callback --------------------
cb = [
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
    callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1),
    callbacks.ModelCheckpoint('best_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
]

# -------------------- 4. Train --------------------
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=120,
    batch_size=32,
    callbacks=cb,
    class_weight=class_weights,
    verbose=2
)

# -------------------- 5. Valutazione --------------------
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n✓ Test accuracy: {acc:.3f}")

# -------------------- 6. Report & Confusion Matrix --------------------
y_pred = model.predict(X_test)
y_pred_labels = np.argmax(y_pred, axis=1)
y_true_labels = np.argmax(y_test, axis=1)

print("\n📊 Classification Report:")
print(classification_report(y_true_labels, y_pred_labels, digits=3))

print("🧩 Confusion Matrix:")
cm = confusion_matrix(y_true_labels, y_pred_labels)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()

# -------------------- 7. Salva il modello finale --------------------
model.save(MODEL_DIR / "lightcurve_cnn_advanced_final.keras")
print("✓ Modello finale salvato come 'lightcurve_cnn_advanced_final.keras'")
