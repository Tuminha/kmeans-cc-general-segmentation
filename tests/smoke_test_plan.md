# Smoke Test Plan

## T1: 00 downloads CC GENERAL to data/raw
- Verify file exists at `data/raw/CC GENERAL.csv`
- Check file size > 1MB

## T2: 01 produces cleaned X_ready with no NA
- Load cleaned data
- Assert no missing values
- Verify shape is reasonable

## T3: 02 reports k grid metrics and picks chosen_k with rationale
- Check that metrics arrays exist for k=2..12
- Verify chosen_k is printed with justification

## T4: 03 saves model, adds labels, prints profile table
- Verify `artifacts/models/kmeans.joblib` exists
- Check DataFrame has 'cluster' column
- Confirm profile table is printed

## T5: 04 prints mean ARI and minibatch comparison table
- Verify bootstrap stability function runs
- Check ARI mean is printed
- Confirm minibatch comparison table exists

## T6: 05 saves PCA plot and a one-page brief
- Verify `artifacts/reports/pca_clusters.png` exists
- Check `artifacts/reports/brief.md` exists with > 12 lines

