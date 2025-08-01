# 05_clustering_analysis.py

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage

# -----------------------------
# 1. Load PCA-reduced data
# -----------------------------
df = pd.read_csv("data/pca_transformed.csv")
X = df.drop("target", axis=1)

os.makedirs("results", exist_ok=True)

# -----------------------------
# 2. Elbow Method for K
# -----------------------------
wcss = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Save Elbow Plot
plt.figure()
plt.plot(K_range, wcss, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('WCSS')
plt.grid(True)
plt.savefig("results/kmeans_elbow_plot.png")
plt.close()

# -----------------------------
# 3. Apply K-Means with K=2
# -----------------------------
kmeans = KMeans(n_clusters=2, random_state=42)
df["kmeans_cluster"] = kmeans.fit_predict(X)

# -----------------------------
# 4. Apply Hierarchical Clustering
# -----------------------------
hierarchical = AgglomerativeClustering(n_clusters=2)
df["hierarchical_cluster"] = hierarchical.fit_predict(X)

# -----------------------------
# 5. Plot Clustering Results
# -----------------------------
# K-Means Clusters
plt.figure()
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='kmeans_cluster', palette='Set2')
plt.title("K-Means Clusters")
plt.savefig("results/kmeans_clusters.png")
plt.close()

# Hierarchical Clusters
plt.figure()
sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='hierarchical_cluster', palette='Set1')
plt.title("Hierarchical Clusters")
plt.savefig("results/hierarchical_clusters.png")
plt.close()

# -----------------------------
# 6. Dendrogram
# -----------------------------
linked = linkage(X, method='ward')

plt.figure(figsize=(10, 6))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=False)
plt.title("Hierarchical Clustering Dendrogram")
plt.savefig("results/hierarchical_dendrogram.png")
plt.close()

print("✅ Clustering complete. Plots saved in 'results/' folder.")
