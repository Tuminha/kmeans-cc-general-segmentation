from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans
from sklearn.utils import resample
from sklearn.metrics import adjusted_rand_score, jaccard_score


def evaluate_clustering(X, labels, kmeans_model):
    """
    Return dict with inertia, silhouette, calinski_harabasz, davies_bouldin.
    
    Parameters:
    -----------
    X : array-like
        The data that was clustered
    labels : array-like
        Cluster labels assigned by the model
    kmeans_model : KMeans
        Fitted KMeans model (to get inertia from)
    
    Returns:
    --------
    dict with keys: 'inertia', 'silhouette', 'calinski_harabasz', 'davies_bouldin'
    """
    inertia = kmeans_model.inertia_
    silhouette = silhouette_score(X, labels)
    calinski_harabasz = calinski_harabasz_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    return {
        'inertia': inertia,
        'silhouette': silhouette,
        'calinski_harabasz': calinski_harabasz,
        'davies_bouldin': davies_bouldin
    }


def bootstrap_stability(X, labels_ref, k, n_boot=20, sample_frac=0.8, random_state=42):
    """
    Refit KMeans on bootstraps and compute ARI vs reference labels.
    
    Parameters:
    -----------
    X : array-like
        Full dataset (n_samples, n_features)
    labels_ref : array-like
        Reference cluster labels for the full dataset (length n_samples)
    k : int
        Number of clusters
    n_boot : int
        Number of bootstrap iterations
    sample_frac : float
        Fraction of samples to use in each bootstrap (default 0.8)
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    ari_scores : list
        List of Adjusted Rand Index scores for each bootstrap
    """
    import numpy as np
    ari_scores = []
    
    # Set random seed
    rng = np.random.RandomState(random_state)
    
    for i in range(n_boot):
        # Get bootstrap sample indices (with replacement)
        n_samples = int(len(X) * sample_frac)
        boot_indices = rng.choice(len(X), size=n_samples, replace=True)
        
        # Get bootstrap sample
        X_boot = X[boot_indices]
        labels_ref_boot = labels_ref[boot_indices]  # Reference labels for bootstrap samples
        
        # Fit KMeans on bootstrap sample
        kmeans_boot = KMeans(n_clusters=k, n_init='auto', random_state=random_state + i)
        kmeans_boot.fit(X_boot)
        labels_boot = kmeans_boot.labels_
        
        # Compute ARI: compare bootstrap labels with reference labels for same samples
        # Both arrays now have the same length (n_samples)
        ari = adjusted_rand_score(labels_ref_boot, labels_boot)
        ari_scores.append(ari)
    
    return ari_scores

