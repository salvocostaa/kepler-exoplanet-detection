# Kepler Exoplanet Detection

Machine learning pipeline for detecting exoplanets using light curves from the NASA Kepler mission.

## 📌 Overview

This project implements a complete pipeline to detect potential exoplanets by analyzing stellar light curves.  
It includes:

- ✅ Data preprocessing and cleaning (light curve extraction)
- ✅ Training a Convolutional Neural Network (CNN) on labeled Kepler data
- ✅ Inference on real-world unseen data and result visualization

The model aims to classify whether a light curve likely contains an exoplanet transit signal.

---

## 📁 Dataset

- Source: [NASA Kepler Exoplanet Search Results (Kaggle)](https://www.kaggle.com/datasets/nasa/kepler-exoplanet-search-results)  
- A subset of the original data is included in the `data/` folder for demonstration purposes.

---

## 📂 Project Structure

```
kepler-exoplanet-detection/
├── data/             # Raw and processed input data
├── src/              # Source code: preprocessing, training, prediction
├── results/          # Output plots, predictions
├── models/           # Saved trained models
├── README.md         # This file
├── requirements.txt  # Dependencies
```

---

## ⚙️ Usage

### 1. Preprocess the dataset

```bash
python src/preprocess_data.py --input data/cumulative.csv --output data/processed
```

### 2. Train the model

```bash
python src/train_model.py
```

- Trains a CNN on cleaned light curves
- Saves the model as `lightcurve_cnn_advanced_final.keras`

### 3. Run predictions on unseen/real data

```bash
python src/predict_exoplanets.py
```

- Loads real light curves from `data/`
- Applies the trained model
- Saves predictions to `predictions_real.csv`
- Generates visual plots in `plots_real/`

---

## 🧪 Output Example

`predictions_real.csv` preview:

| ID         | Prediction | Probability |
|------------|------------|-------------|
| kplr_00001 | 1          | 0.94        |
| kplr_00002 | 0          | 0.17        |

Each prediction is visualized as a `.png` plot showing the model's decision.

![Esempio di plot](results/plot.png)

---

## 🧰 Requirements

Install dependencies via:

```bash
pip install -r requirements.txt
```

Key libraries:

- `tensorflow` / `keras`
- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`

---

## 📖 References

- NASA Kepler Project & Kaggle Dataset  
- [GeeksforGeeks - CNNs, RNNs, and Machine Learning](https://www.geeksforgeeks.org/)  
- Università Bocconi – Neural Network Introduction  

---

## 📄 License

This project is licensed under the MIT License.
