import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy import stats


def plot_metric_vs_k(k_grid, sil, ch, db, inertia):
    """Plot four metrics across k (2..12)."""
    # Ensure all inputs are numpy arrays and convert to float
    k_grid = np.asarray(k_grid, dtype=float).flatten()
    sil = np.asarray(sil, dtype=float).flatten()
    ch = np.asarray(ch, dtype=float).flatten()
    db = np.asarray(db, dtype=float).flatten()
    inertia = np.asarray(inertia, dtype=float).flatten()
    
    # Verify all arrays have the same length
    if not (len(k_grid) == len(sil) == len(ch) == len(db) == len(inertia)):
        raise ValueError(f"All arrays must have the same length. Got: k={len(k_grid)}, sil={len(sil)}, ch={len(ch)}, db={len(db)}, inertia={len(inertia)}")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    # Plot each metric with the correct data - use positional arguments (not x= and y=)
    axes[0].plot(k_grid, sil, marker='o', linestyle='-')
    axes[0].set_title('Silhouette Score')
    axes[0].set_xlabel('k')
    axes[0].set_ylabel('Silhouette Score')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(k_grid, ch, marker='o', linestyle='-')
    axes[1].set_title('Calinski Harabasz Score')
    axes[1].set_xlabel('k')
    axes[1].set_ylabel('Calinski Harabasz Score')
    axes[1].grid(True, alpha=0.3)
    
    # Invert Davies-Bouldin (lower is better) by plotting negative values
    db_inverted = -db  # Use numpy array operation instead of list comprehension
    axes[2].plot(k_grid, db_inverted, marker='o', linestyle='-')
    axes[2].set_title('Davies Bouldin Score (inverted)')
    axes[2].set_xlabel('k')
    axes[2].set_ylabel('Davies Bouldin Score (inverted)')
    axes[2].grid(True, alpha=0.3)
    
    axes[3].plot(k_grid, inertia, marker='o', linestyle='-')
    axes[3].set_title('Inertia')
    axes[3].set_xlabel('k')
    axes[3].set_ylabel('Inertia')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure to images directory
    os.makedirs('../images', exist_ok=True)
    plt.savefig('../images/metric_vs_k.png', dpi=150, bbox_inches='tight')
    plt.show()

def cluster_profile_table(df_with_labels, feature_list, label_col="cluster"):
    """Return a tidy profile table: per-cluster mean, z-score, size."""
    # Calculate cluster sizes
    cluster_sizes = df_with_labels.groupby(label_col).size().to_dict()
    
    # Calculate per-cluster means for specified features
    cluster_means = df_with_labels.groupby(label_col)[feature_list].mean()
    
    # Calculate z-scores: (cluster_mean - overall_mean) / overall_std
    overall_means = df_with_labels[feature_list].mean()
    overall_stds = df_with_labels[feature_list].std()
    
    cluster_zscores = pd.DataFrame(index=cluster_means.index, columns=cluster_means.columns)
    for feature in feature_list:
        cluster_zscores[feature] = (cluster_means[feature] - overall_means[feature]) / overall_stds[feature]
    
    # Create a comprehensive profile table
    profile = pd.DataFrame()
    for cluster in cluster_means.index:
        cluster_data = {
            'cluster': cluster,
            'size': cluster_sizes[cluster]
        }
        # Add means
        for feature in feature_list:
            cluster_data[f'{feature}_mean'] = cluster_means.loc[cluster, feature]
            cluster_data[f'{feature}_zscore'] = cluster_zscores.loc[cluster, feature]
        profile = pd.concat([profile, pd.DataFrame([cluster_data])], ignore_index=True)
    
    return profile
