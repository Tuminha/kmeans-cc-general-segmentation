def plot_metric_vs_k(k_grid, sil, ch, db, inertia):
    """Plot four metrics across k (2..12)."""
    # TODO: matplotlib subplots; invert DB (lower is better) visually
    raise NotImplementedError

def cluster_profile_table(df_with_labels, feature_list, label_col="cluster"):
    """Return a tidy profile table: per-cluster mean, z-score, size."""
    # TODO: groupby, describe, z-scores by cluster
    raise NotImplementedError

