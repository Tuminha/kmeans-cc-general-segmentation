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
│   ├── 01_eda_preprocess.ipynb    ✅ Complete
│   ├── 02_k_selection_silhouette_ch_db.ipynb
│   ├── 03_fit_kmeans_and_profile.ipynb
│   ├── 04_stability_and_minibatch.ipynb
│   └── 05_pca_visualize_and_brief.ipynb
├── src/                   # Utility modules
└── tests/                 # Test plans
```

## Deliverables

- ✅ **Notebook 00**: Data download from Kaggle (complete)
- ✅ **Notebook 01**: EDA and preprocessing (complete)
- **Notebook 02**: k selection with elbow, Silhouette, CH, DB + majority vote
- **Notebook 03**: trained KMeans model, labeled dataset, profiles (size, spend, z-score radar)
- **Notebook 04**: stability (bootstrapped ARI/Jaccard), MiniBatchKMeans speed/quality comparison
- **Notebook 05**: PCA/UMAP 2D plots and a one-page brief in `artifacts/reports/`

## Preprocessing Results (Notebook 01)

### Data Cleaning Pipeline
1. **Outlier Clipping**: Winsorized at 1st and 99th percentiles
2. **Log Transformation**: Applied `log1p()` to all numeric features to handle right-skewed distributions
3. **Missing Value Imputation**: Median imputation for all features

### Key Findings

**Skewness Reduction** (critical for K-Means clustering):
- **MINIMUM_PAYMENTS**: 13.62 → 0.36 (97% reduction) ✅
- **PURCHASES**: 8.14 → -0.78 (excellent improvement) ✅
- **ONEOFF_PAYMENTS**: 10.05 → 0.18 (98% reduction) ✅

**Skewness Interpretation**:
- **Before cleaning**: Multiple features with extreme skewness (>5.0), making them unsuitable for K-Means
- **After cleaning**: All features now have skewness between -1.0 and +1.0, with most in the excellent range (-0.5 to +0.5)
- **Result**: Data is now well-suited for K-Means clustering, which assumes spherical clusters

**Missing Values**:
- `MINIMUM_PAYMENTS`: Most missing values (>3% of dataset)
- All missing values handled via median imputation after transformation

### Quality Checks
- ✅ No missing values remain after cleaning
- ✅ All features have acceptable skewness for clustering
- ✅ Data standardized and ready for K-Means

### Correlation Analysis

The correlation heatmap reveals important relationships between features that inform clustering decisions:

**Strong Positive Correlations** (indicating feature groups):
- **Purchase-related features**: `PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES`, `PURCHASES_FREQUENCY`, and `PURCHASES_TRX` are highly correlated (0.8-0.9), suggesting customers who engage in one type of purchase behavior tend to engage in others.
- **Cash Advance features**: `CASH_ADVANCE`, `CASH_ADVANCE_FREQUENCY`, and `CASH_ADVANCE_TRX` show strong positive correlations (0.7-0.8), indicating a distinct customer segment.
- **Balance and Credit**: `BALANCE` and `CREDIT_LIMIT` correlate at 0.53, showing that higher credit limits often correspond to higher outstanding balances.

**Weak/Negative Correlations** (indicating distinct segments):
- **Cash Advance vs. Purchases**: Weak negative correlation (-0.05) suggests customers who use cash advances are a different segment from frequent purchasers.
- **Payment behavior**: `PRC_FULL_PAYMENT` shows negative correlation with `BALANCE` (-0.32), indicating customers who pay in full tend to have lower balances.

**Clustering Implications**:
- High correlations suggest potential redundancy, which could benefit from dimensionality reduction (PCA) in future iterations.
- Distinct correlation blocks (purchase-related vs. cash advance-related) hint at natural customer segments that K-Means should discover.

<div align="center">

<img src="images/correlation_heatmap.png" alt="Correlation Heatmap of Credit Card Features" width="800" />

*Correlation heatmap showing relationships between 18 credit card usage features*

</div>

## How to run

1) `pip install -r requirements.txt`

2) ✅ Data is already downloaded in `data/raw/CC GENERAL.csv` (via Notebook 00)

3) Execute notebooks 01 → 05 in order.

