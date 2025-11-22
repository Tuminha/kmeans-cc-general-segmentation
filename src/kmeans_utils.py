def evaluate_clustering(X, labels):
    """
    Return dict with inertia, silhouette, calinski_harabasz, davies_bouldin.
    # TODO: use sklearn metrics
    """
    raise NotImplementedError

def bootstrap_stability(X, labels_ref, k, n_boot=20, sample_frac=0.8, random_state=42):
    """
    Refit KMeans on bootstraps and compute ARI/Jaccard vs reference labels.
    # TODO: sklearn.utils.resample, sklearn.metrics.adjusted_rand_score
    """
    raise NotImplementedError

