import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectKBest
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv("data/cleaned_heart.csv")


X = df.drop("target", axis=1)
y = df["target"].astype(int)  

import os
os.makedirs("results", exist_ok=True)


model_rf = RandomForestClassifier(random_state=42)
model_rf.fit(X, y)
importances = model_rf.feature_importances_


feature_importance_df = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)


plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_importance_df)
plt.title("Feature Importance from Random Forest")
plt.tight_layout()
plt.savefig("results/feature_importance_rf.png")
plt.close()


model_lr = LogisticRegression(max_iter=1000)
rfe = RFE(estimator=model_lr, n_features_to_select=5)
rfe.fit(X, y)

rfe_selected_features = X.columns[rfe.support_]
print("\n✅ Top 5 RFE Selected Features:")
print(list(rfe_selected_features))


scaler = MinMaxScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

chi2_selector = SelectKBest(score_func=chi2, k=5)
chi2_selector.fit(X_scaled, y)
chi2_selected_features = X.columns[chi2_selector.get_support()]

print("\n✅ Top 5 Chi-Square Selected Features:")
print(list(chi2_selected_features))


selected_features = set(rfe_selected_features) | set(chi2_selected_features) | set(feature_importance_df["Feature"][:5])
selected_features = list(selected_features)


X_selected = X[selected_features]
X_selected["target"] = y
X_selected.to_csv("data/selected_features.csv", index=False)

print(f"\n✅ Total Selected Features Saved: {len(selected_features)}")
print("💾 Reduced dataset saved as 'data/selected_features.csv'")
