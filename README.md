# K-Means Customer Segmentation — CC GENERAL

## Goal

- Learn K-Means end-to-end on the classic **Credit Card Dataset for Clustering (CC GENERAL)**, 
  selecting k with **Silhouette**, **Calinski–Harabasz (CH)**, and **Davies–Bouldin (DB)**,
  validating stability, and producing manager-friendly cluster profiles.

## Data

- Kaggle: **Credit Card Dataset for Clustering (CC GENERAL)** by Arjun Bhasin.
  File: `CC GENERAL.csv` (29k rows, 18 numeric features; anonymized credit-card usage).
- **Status**: ✅ Data downloaded and available in `data/raw/CC GENERAL.csv` (0.86 MB)

## Why this dataset

- Clean numeric features, strong real-world segmentation signal (spend, payments, balances).

## Project Structure

```
kmeans-cc-general-segmentation/
├── data/
│   ├── raw/               # CC GENERAL.csv (downloaded)
│   └── interim/           # Processed data
├── artifacts/
│   ├── models/            # Saved KMeans models
│   └── reports/           # Visualizations and briefs
├── notebooks/
│   ├── 00_get_data.ipynb          ✅ Complete
│   ├── 01_eda_preprocess.ipynb    🔄 In progress
│   ├── 02_k_selection_silhouette_ch_db.ipynb
│   ├── 03_fit_kmeans_and_profile.ipynb
│   ├── 04_stability_and_minibatch.ipynb
│   └── 05_pca_visualize_and_brief.ipynb
├── src/                   # Utility modules
└── tests/                 # Test plans
```

## Deliverables

- ✅ **Notebook 00**: Data download from Kaggle (complete)
- **Notebook 01**: EDA and preprocessing (in progress)
- **Notebook 02**: k selection with elbow, Silhouette, CH, DB + majority vote
- **Notebook 03**: trained KMeans model, labeled dataset, profiles (size, spend, z-score radar)
- **Notebook 04**: stability (bootstrapped ARI/Jaccard), MiniBatchKMeans speed/quality comparison
- **Notebook 05**: PCA/UMAP 2D plots and a one-page brief in `artifacts/reports/`

## How to run

1) `pip install -r requirements.txt`

2) ✅ Data is already downloaded in `data/raw/CC GENERAL.csv` (via Notebook 00)

3) Execute notebooks 01 → 05 in order.

