#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score, homogeneity_score, completeness_score, v_measure_score, silhouette_scor

Class KMeans:
    def __init__(self, n_clusters=3, max_iters=100, random_state=None, init='random'):
        """
        Initialize KMeans clustering algorithm

        Parameters:
        -----------
        n_clusters : int, default=3
            Number of clusters (K value)
        max_iters : int, default=100
            Maximum number of iterations
        random_state : int, default=None
            Random seed for reproducibility
        init : str, default='random'
            Method for initialization, 'random' or 'kmeans++'
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.init = init
        self.centroids = None
        self.labels = None
        self.inertia = None
        self.n_iter_ = 0  # Number of iterations run

    def fit(self, X):
        """
        Fit KMeans to the data

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data

        Returns:
        --------
        self : object
            Fitted estimator
        """
        # Set random seed for reproducibility
        if self.random_state is not None:
            np.random.seed(self.random_state)

        # Get dimensions of input data
        n_samples, n_features = X.shape

        # Initialize centroids based on chosen method
        if self.init == 'random':
            # Initialize centroids by randomly selecting data points
            idx = np.random.choice(n_samples, self.n_clusters, replace=False)
            self.centroids = X[idx]
        elif self.init == 'kmeans++':
            # Initialize using k-means++ algorithm
            self.centroids = self._kmeans_plus_plus_init(X)
        else:
            raise ValueError("init should be either 'random' or 'kmeans++'")

        # Initialize previous centroids for convergence check
        prev_centroids = np.zeros_like(self.centroids)

        # Initialize cluster assignments
        self.labels = np.zeros(n_samples)

        # Main loop
        for iteration in range(self.max_iters):
            # Check for convergence
            if np.all(self.centroids == prev_centroids):
                break

            # Store current centroids for convergence check
            prev_centroids = self.centroids.copy()

            # Assign each data point to the nearest centroid
            self.labels = self._assign_clusters(X)

            # Update centroids based on new cluster assignments
            self._update_centroids(X)

            # Update iteration count
            self.n_iter_ = iteration + 1

        # Calculate inertia (sum of squared distances to centroids)
        self.inertia = self._calculate_inertia(X)

        return self

    def _kmeans_plus_plus_init(self, X):
        """
        Initialize centroids using k-means++ algorithm

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            Training data

        Returns:
        --------
        centroids : array of shape (n_clusters, n_features)
            Initial centroids
        """
        n_samples, n_features = X.shape
        centroids = np.zeros((self.n_clusters, n_features))

        # Choose first centroid randomly
        first_idx = np.random.choice(n_samples)
        centroids[0] = X[first_idx]

        # Choose remaining centroids
        for i in range(1, self.n_clusters):
            # Calculate distances from points to nearest existing centroid
            min_dists = np.min([np.sum((X - cent) ** 2, axis=1) for cent in centroids[:i]], axis=0)

            # Choose next centroid with probability proportional to squared distance
            probs = min_dists / min_dists.sum()
            next_idx = np.random.choice(n_samples, p=probs)
            centroids[i] = X[next_idx]

        return centroids

    def _assign_clusters(self, X):
        """Assign each data point to the nearest centroid"""
        distances = np.zeros((X.shape[0], self.n_clusters))

        for i, centroid in enumerate(self.centroids):
            # Calculate Euclidean distance from each point to the centroid
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))

        # Return index of the closest centroid for each point
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X):
        """Update centroids based on mean of assigned points"""
        for i in range(self.n_clusters):
            # Get points assigned to the current cluster
            cluster_points = X[self.labels == i]

            # Update centroid if the cluster is not empty
            if len(cluster_points) > 0:
                self.centroids[i] = np.mean(cluster_points, axis=0)

    def _calculate_inertia(self, X):
        """Calculate inertia (sum of squared distances to assigned centroids)"""
        inertia = 0
        for i in range(self.n_clusters):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[i]) ** 2)
        return inertia

    def predict(self, X):
        """
        Predict the closest cluster for each sample in X

        Parameters:
        -----------
        X : array-like of shape (n_samples, n_features)
            New data to predict

        Returns:
        --------
        labels : array of shape (n_samples,)
            Index of the cluster each sample belongs to
        """
        return self._assign_clusters(X)

def elbow_method(X, max_k=10):
    """
    Perform elbow method to find optimal k

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Training data
    max_k : int, default=10
        Maximum number of clusters to try
    """
    inertias = []

    for k in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        inertias.append(kmeans.inertia)

def evaluate_metrics(kmeans, X, true_labels=None):
    """
    Evaluate clustering using various metrics

    Parameters:
    -----------
    kmeans : KMeans object
        Fitted KMeans model
    X : array-like of shape (n_samples, n_features)
        Data points
    true_labels : array of shape (n_samples,), optional
        True cluster labels, if available for comparison

    Returns:
    --------
    metrics : dict
        Dictionary containing all calculated metrics
    """
    metrics = {}
    labels = kmeans.labels

    # Inertia (sum of squared distances)
    metrics['inertia'] = kmeans.inertia

    # Silhouette Score
    try:
        metrics['silhouette_score'] = silhouette_score(X, labels)
    except:
        metrics['silhouette_score'] = None

    # Metrics that require true labels
    if true_labels is not None:
        # Adjusted Rand Index
        metrics['adjusted_rand_index'] = adjusted_rand_score(true_labels, labels)

        # Homogeneity
        metrics['homogeneity'] = homogeneity_score(true_labels, labels)

        # Completeness
        metrics['completeness'] = completeness_score(true_labels, labels)

        # V-Measure
        metrics['v_measure'] = v_measure_score(true_labels, labels)

    return metrics


def normalize_data(X, method='standard'):
    """
    Normalize the data using various methods

    Parameters:
    -----------
    X : array-like of shape (n_samples, n_features)
        Data to normalize
    method : str, default='standard'
        Normalization method: 'standard', 'minmax', or 'robust'

    Returns:
    --------
    X_normalized : array-like of shape (n_samples, n_features)
        Normalized data
    """
    try:
        if method == 'standard':
            # Standardization (z-score): (x - mean) / std
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_normalized = scaler.fit_transform(X)
        elif method == 'minmax':
            # Min-max scaling: (x - min) / (max - min)
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            X_normalized = scaler.fit_transform(X)
        elif method == 'robust':
            # Robust scaling: (x - median) / IQR
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            X_normalized = scaler.fit_transform(X)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return X_normalized
    except ImportError:
        print("Scikit-learn not available for normalization. Using manual standardization...")
        # Manual standardization
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std[std == 0] = 1  # Avoid division by zero
        return (X - mean) / std

