# K-Means Customer Segmentation — CC GENERAL

## Goal

- Learn K-Means end-to-end on the classic **Credit Card Dataset for Clustering (CC GENERAL)**, 
  selecting k with **Silhouette**, **Calinski–Harabasz (CH)**, and **Davies–Bouldin (DB)**,
  validating stability, and producing manager-friendly cluster profiles.

## Data

- Kaggle: **Credit Card Dataset for Clustering (CC GENERAL)** by Arjun Bhasin.
  File: `CC GENERAL.csv` (29k rows, 18 numeric features; anonymized credit-card usage).

## Why this dataset

- Clean numeric features, strong real-world segmentation signal (spend, payments, balances).

## Deliverables

- Notebook 02: k selection with elbow, Silhouette, CH, DB + majority vote.
- Notebook 03: trained KMeans model, labeled dataset, profiles (size, spend, z-score radar).
- Notebook 04: stability (bootstrapped ARI/Jaccard), MiniBatchKMeans speed/quality comparison.
- Notebook 05: PCA/UMAP 2D plots and a one-page brief in `artifacts/reports/`.

## How to run

1) `pip install -r requirements.txt`

2) Use Notebook 00 to download from Kaggle (or drop `CC GENERAL.csv` into `data/raw/`)

3) Execute notebooks 01 → 05 in order.

