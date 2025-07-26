#!/usr/bin/env python
"""
preprocess.py  |  Kepler light‑curve preprocessing avanzato
--------------------------------------------------------------
1.  legge il CSV prodotto da build_lightcurves_csv_balanced.py
2.  rimuove NaN/Inf
3.  DETREND   : rolling‑median (finestra 101)   -> trend‑free flux
4.  SCALING   : z‑score per curva (mean 0, std 1)
5.  oversample bilanciato: porta tutte le classi allo stesso numero
6.  pad/crop   : porta tutte le curve a MAX_LEN
7.  salva X e y in formato NumPy
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import median_filter
from sklearn.model_selection import train_test_split
from collections import Counter
from tqdm import tqdm

# ---------------- parametri globali ----------------
MAX_LEN    = 512
FLUX_COL   = "flux"
ID_COL     = "curve_id"
LABEL_COL  = "label"        # PC / FP / EB
SEED       = 42
# ---------------------------------------------------

np.random.seed(SEED)


def load_df(csv_path: Path) -> pd.DataFrame:
    print(f"Caricamento CSV da: {csv_path}")
    df = pd.read_csv(csv_path)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(subset=[FLUX_COL, ID_COL, LABEL_COL], inplace=True)
    return df


def detrend_flux(flux: np.ndarray, win: int = 101) -> np.ndarray:
    trend = median_filter(flux, size=win, mode="nearest")
    return flux - trend


def zscore(seq: np.ndarray) -> np.ndarray:
    mu  = np.mean(seq)
    std = np.std(seq) + 1e-6
    return (seq - mu) / std


def pad_or_crop(seq: np.ndarray, target_len: int = MAX_LEN) -> np.ndarray:
    if len(seq) >= target_len:
        return seq[:target_len]
    return np.pad(seq, (0, target_len - len(seq)), mode="constant")


def build_arrays(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for cid, group in tqdm(df.groupby(ID_COL), desc="Curve"):
        flux = group[FLUX_COL].values
        flux = detrend_flux(flux)
        flux = zscore(flux)
        flux = pad_or_crop(flux)
        assert len(flux) == MAX_LEN, f"Errore: curva {cid} ha lunghezza {len(flux)}"
        X.append(flux)
        y.append(group[LABEL_COL].iloc[0])
    return np.stack(X).astype("float32"), np.array(y)


def oversample_balanced(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    unique_classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_out, y_out = [X], [y]

    for cls, count in zip(unique_classes, counts):
        if count < max_count:
            n_dup = max_count - count
            mask = y == cls
            dup_idx = np.random.choice(np.where(mask)[0], n_dup, replace=True)
            X_out.append(X[dup_idx])
            y_out.append(y[dup_idx])

    X_bal = np.concatenate(X_out)
    y_bal = np.concatenate(y_out)

    print(f"Distribuzione dopo oversampling bilanciato: {dict(zip(*np.unique(y_bal, return_counts=True)))}")
    return X_bal, y_bal


def encode_labels(labels: np.ndarray) -> np.ndarray:
    mapping = {"PC": 0, "FP": 1, "EB": 2}
    return np.array([mapping[label] for label in labels])


def main(args):
    print("➜  Carico CSV…")
    df = load_df(args.input)

    print("➜  Detrending + z‑score + resize…")
    X, y_raw = build_arrays(df)

    print("➜  Oversampling bilanciato delle classi…")
    X, y_raw = oversample_balanced(X, y_raw)

    print("➜  Distribuzione classi (dopo oversampling):")
    print(dict(Counter(y_raw)))

    print("➜  Encoding label e split train/test…")
    y = encode_labels(y_raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "X_train.npy", X_train)
    np.save(out / "X_test.npy",  X_test)
    np.save(out / "y_train.npy", y_train)
    np.save(out / "y_test.npy",  y_test)

    print("✓ Salvataggio completato")
    print("Shape X_train:", X_train.shape, " y_train:", y_train.shape)


if __name__ == "__main__":
    import os
    from pathlib import Path

    ROOT_DIR = Path(__file__).parent.parent.resolve()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT_DIR / "data" / "lightcurves.csv",
        help="CSV da preprocessare"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "data" / "processed",
        help="Cartella di destinazione .npy"
    )
    args = parser.parse_args()
    main(args)


