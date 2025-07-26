import pandas as pd
from pathlib import Path

ROOT_DIR = Path(__file__).parent.parent.resolve()
DATA_DIR = ROOT_DIR / "data"

TRAIN_CSV = DATA_DIR / "lightcurves.csv"
TEST_CSV = DATA_DIR / "lightcurves_test.csv"
TRAIN_CLEAN_CSV = DATA_DIR / "lightcurves_train_clean.csv"

N_PER_CLASS = 5

# Leggi tutto il dataset di training
df = pd.read_csv(TRAIN_CSV)

# Verifica che ci siano le colonne 'label' e 'curve_id'
assert "label" in df.columns, "Manca la colonna 'label' nel CSV"
assert "curve_id" in df.columns, "Manca la colonna 'curve_id' nel CSV"

# Prendi curve uniche per classe (evita duplicati di curve_id)
unique_curves = df.drop_duplicates(subset=["curve_id"])

# Filtra solo le classi desiderate
classi_target = ["PC", "FP", "EB"]
unique_curves = unique_curves[unique_curves["label"].isin(classi_target)]

# Campiona N curve per ciascuna classe per il test
samples = []
for cls in classi_target:
    cls_df = unique_curves[unique_curves["label"] == cls]
    if len(cls_df) < N_PER_CLASS:
        raise ValueError(f"⚠️ Classe '{cls}' ha solo {len(cls_df)} curve disponibili, impossibile campionarne {N_PER_CLASS}.")
    samples.append(cls_df.sample(n=N_PER_CLASS, random_state=42))

# Combina le curve campionate per il test
test_curves = pd.concat(samples)

# Filtra dal dataset originale solo le curve selezionate per il test
test_df = df[df["curve_id"].isin(test_curves["curve_id"])]

# Salva il CSV di test
test_df.to_csv(TEST_CSV, index=False)

print(f"✅ Dataset test creato in '{TEST_CSV}' con {len(test_curves)} curve uniche (PC, FP, EB = {N_PER_CLASS} ciascuna)")

# Ora creo il dataset di train **escludendo** le curve di test
train_clean_df = df[~df["curve_id"].isin(test_curves["curve_id"])]
train_clean_df.to_csv(TRAIN_CLEAN_CSV, index=False)

print(f"✅ Dataset train pulito creato in '{TRAIN_CLEAN_CSV}' senza curve di test")
