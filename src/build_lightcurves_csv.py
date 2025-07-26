#!/usr/bin/env python
"""
build_lightcurves_csv_balanced.py  ─  v3  (PC-focused balancing)
-----------------------------------------------------------------
Scarica curve Kepler, ritaglia 512 punti centrati sul transito
(per PC / EB), esegue detrending + z‑score, salva CSV bilanciato
sfruttando tutte le curve PC disponibili e bilanciando le altre.
"""

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from lightkurve import search_lightcurve
from tqdm import tqdm

# ------------- PARAMETRI ----------------------------------------
BASE_DIR = Path(__file__).parent.parent  # dalla cartella src/ sali a root della repo
DATA_DIR = BASE_DIR / "data"             # cartella data/

CATALOG = DATA_DIR / "cumulative.csv"    # data/cumulative.csv
OUT_CSV = DATA_DIR / "lightcurves.csv"   # catalogo NASA KOI

MAX_LEN        = 512                  # punti per curva
ROLL_WIN       = 101                  # finestra median filter detrend
SEED           = 42
# -----------------------------------------------------------------

warnings.filterwarnings("ignore")
rng_global = np.random.default_rng(SEED)

# -----------------------------------------------------------------
# 1. utilità
# -----------------------------------------------------------------
def disposition_to_label(d: str) -> str:
    if d == "CONFIRMED":
        return "PC"
    if d == "FALSE POSITIVE":
        return "FP"
    return "EB"

def zscore(arr: np.ndarray) -> np.ndarray:
    mu = np.nanmean(arr)
    sd = np.nanstd(arr) + 1e-6
    return (arr - mu) / sd

def detrend(flux: np.ndarray, win: int = ROLL_WIN) -> np.ndarray:
    trend = pd.Series(flux).rolling(win, center=True, min_periods=1).median()
    return flux - trend.values

def already_done(kepid: int) -> bool:
    if not OUT_CSV.exists():
        return False
    try:
        done = pd.read_csv(OUT_CSV, usecols=["curve_id"])["curve_id"].values
        return kepid in done
    except Exception:
        return False

# -----------------------------------------------------------------
# 2. carica metadati (periodo ed epoca) in un dizionario
# -----------------------------------------------------------------
CAT_COLS = ["kepid", "koi_disposition", "koi_period", "koi_time0bk"]
meta = (pd.read_csv(CATALOG, usecols=CAT_COLS)
          .set_index("kepid"))

# -----------------------------------------------------------------
# 3. download + ritaglio
# -----------------------------------------------------------------
def process_kepid(kepid: int, label: str):
    try:
        coll = search_lightcurve(f"KIC {kepid}", mission="Kepler").download_all()
        if not coll:
            return None

        lc = coll.stitch().remove_nans()

        # ------------ finestra centrata -------------
        per = meta.at[kepid, "koi_period"]
        t0  = meta.at[kepid, "koi_time0bk"]

        if label in ("PC", "EB") and pd.notna(per) and pd.notna(t0):
            folded = lc.fold(per, t0=t0).remove_nans()
            order  = np.argsort(folded.phase)
            flux   = folded.flux.value[order]
            mid    = len(flux) // 2
            half   = MAX_LEN // 2
            window = flux[mid-half: mid+half]
        else:
            flux_all = lc.flux.value
            if len(flux_all) < MAX_LEN:
                return None
            start = rng_global.integers(0, len(flux_all) - MAX_LEN + 1)
            window = flux_all[start:start+MAX_LEN]

        if len(window) < MAX_LEN:
            return None

        window = detrend(window)
        window = zscore(window)

        return pd.DataFrame({
            "curve_id": kepid,
            "flux": window,
            "label": label
        })

    except Exception as e:
        print(f"⚠️ {kepid}: {e}")
        return None

# -----------------------------------------------------------------
# 4. main
# -----------------------------------------------------------------
def main():
    cat = meta.reset_index()[["kepid", "koi_disposition"]]
    cat["label"] = cat["koi_disposition"].apply(disposition_to_label)

    # Prendi tutte le curve PC
    pc_group = cat[cat["label"] == "PC"]
    print(f"→ Curve PC disponibili: {len(pc_group)}")

    # Campiona stesso numero da EB e FP (senza superare)
    n_target = len(pc_group)
    sel = [pc_group[["kepid", "label"]]]

    for lab in ["FP", "EB"]:
        grp = cat[cat["label"] == lab]
        picks = grp.sample(n=min(n_target, len(grp)), random_state=rng_global.integers(1e9))
        sel.append(picks[["kepid", "label"]])

    tasks_df = pd.concat(sel).reset_index(drop=True)
    print(f"→ Totale curve selezionate: {len(tasks_df)}")

    if OUT_CSV.exists():
        OUT_CSV.unlink()

    counts, processed = {"PC": 0, "FP": 0, "EB": 0}, 0

    for row in tqdm(tasks_df.itertuples(index=False), total=len(tasks_df)):
        if already_done(row.kepid):
            continue
        df_curve = process_kepid(row.kepid, row.label)
        if df_curve is not None:
            header = not OUT_CSV.exists()
            df_curve.to_csv(OUT_CSV, mode="a", header=header, index=False)
            counts[row.label] += 1
            processed += 1

    print(f"\n✓ Salvate {processed} curve in '{OUT_CSV}'")
    print("Distribuzione classi:", counts)

if __name__ == "__main__":
    main()
