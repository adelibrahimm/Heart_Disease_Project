# 01_data_preprocessing.py

import matplotlib
matplotlib.use("Agg")  # ✅ Prevent GUI backend errors

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# إعداد الرسومات
sns.set(style="whitegrid", palette="muted")
plt.rcParams["figure.figsize"] = (10, 6)

# -----------------------------
# 1. تحميل البيانات
# -----------------------------
df = pd.read_csv("data/heart.csv")
print("🔹 First 5 rows:")
print(df.head())
print("\n🔹 Dataset Info:")
print(df.info())
print("\n🔹 Null values per column:")
print(df.isnull().sum())
print("\n🔹 Summary Statistics:")
print(df.describe())

# -----------------------------
# 2. EDA: رسوم بيانية محفوظة
# -----------------------------
# أنشئ مجلد النتائج إذا غير موجود
import os
os.makedirs("results", exist_ok=True)

# Histogram
df.hist(bins=20, edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=16)
plt.tight_layout()
plt.savefig("results/histograms.png")
plt.close()

# Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Correlation Heatmap")
plt.savefig("results/correlation_heatmap.png")
plt.close()

# Boxplots
for col in df.select_dtypes(include=np.number).columns:
    plt.figure()
    sns.boxplot(x=df[col])
    plt.title(f"Boxplot of {col}")
    plt.savefig(f"results/boxplot_{col}.png")
    plt.close()

# -----------------------------
# 3. Scaling
# -----------------------------
scaler = StandardScaler()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
df_scaled = df.copy()
df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])

print("\n✅ Scaling complete. Scaled sample:")
print(df_scaled.head())

# -----------------------------
# 4. حفظ البيانات النظيفة
# -----------------------------
df_scaled.to_csv("data/cleaned_heart.csv", index=False)
print("\n💾 Cleaned and scaled dataset saved as 'data/cleaned_heart.csv'")
