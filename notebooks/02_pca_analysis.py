import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/cleaned_heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

pca = PCA()
X_pca = pca.fit_transform(X)


explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

plt.figure(figsize=(10, 5))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.7, label='Individual')
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', color='red', label='Cumulative')
plt.xlabel('Number of Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Explained Variance by PCA Components')
plt.legend()
plt.grid(True)
plt.savefig("results/pca_variance_plot.png")
plt.close()

plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='coolwarm', edgecolor='k', alpha=0.7)
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.title("PCA - First Two Components")
plt.colorbar(label='Target')
plt.savefig("results/pca_scatter_plot.png")
plt.close()


X_pca_df = pd.DataFrame(X_pca[:, :2], columns=['PCA1', 'PCA2'])
X_pca_df["target"] = y
X_pca_df.to_csv("data/pca_transformed.csv", index=False)

print("✅ PCA complete. Saved 2D PCA data and plots in 'results/' folder.")
