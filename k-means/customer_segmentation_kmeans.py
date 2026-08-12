# ============================================================
# K-Means Clustering Lab — Customer Mall Dataset
# Run this entire file top to bottom in Jupyter Notebook
# Each section produces a screenshot-ready output
# ============================================================

# ── SECTION 1: Import Libraries ──────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')
plt.switch_backend("Agg")

print("✅ All libraries imported successfully!")
print(f"pandas   : {pd.__version__}")
print(f"numpy    : {np.__version__}")
print(f"seaborn  : {sns.__version__}")

# ── SECTION 2: Create / Load Dataset ─────────────────────────
# Synthesized Mall Customer dataset (200 customers)
np.random.seed(42)
n = 200

customer_id   = np.arange(1, n + 1)
gender        = np.random.choice(['Male', 'Female'], n)
age           = np.random.randint(18, 70, n)
annual_income = np.concatenate([
    np.random.normal(30,  5, 40),   # low income
    np.random.normal(55,  8, 60),   # mid income
    np.random.normal(85, 10, 60),   # high income
    np.random.normal(70,  6, 40),   # mid-high
])
annual_income = np.clip(annual_income, 15, 140).astype(int)

spending_score = np.concatenate([
    np.random.normal(70, 10, 40),   # low income → high spending
    np.random.normal(50,  8, 60),   # mid income → avg spending
    np.random.normal(25,  8, 60),   # high income → low spending
    np.random.normal(75, 10, 40),   # mid-high income → high spending
])
spending_score = np.clip(spending_score, 1, 100).astype(int)

df = pd.DataFrame({
    'CustomerID'    : customer_id,
    'Gender'        : gender,
    'Age'           : age,
    'AnnualIncome'  : annual_income,
    'SpendingScore' : spending_score,
})

print("\n── Dataset Shape ──")
print(f"Rows: {df.shape[0]}  |  Columns: {df.shape[1]}")
print("\n── First 10 Rows ──")
print(df.head(10).to_string(index=False))

# ── SECTION 3: Dataset Audit ──────────────────────────────────
print("\n── Data Types ──")
print(df.dtypes)

print("\n── Missing Values ──")
print(df.isnull().sum())

print("\n── Descriptive Statistics ──")
print(df.describe().round(2))

print("\n── Gender Distribution ──")
print(df['Gender'].value_counts())

# ── SECTION 4: Exploratory Data Analysis ─────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("EDA — Feature Distributions", fontsize=14, fontweight='bold')

axes[0].hist(df['Age'], bins=20, color='steelblue', edgecolor='white')
axes[0].set_title("Age Distribution")
axes[0].set_xlabel("Age"); axes[0].set_ylabel("Count")

axes[1].hist(df['AnnualIncome'], bins=20, color='seagreen', edgecolor='white')
axes[1].set_title("Annual Income (k$)")
axes[1].set_xlabel("Income (k$)"); axes[1].set_ylabel("Count")

axes[2].hist(df['SpendingScore'], bins=20, color='tomato', edgecolor='white')
axes[2].set_title("Spending Score (1–100)")
axes[2].set_xlabel("Score"); axes[2].set_ylabel("Count")

plt.tight_layout()
plt.savefig("eda_distributions.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ EDA plot saved as eda_distributions.png")

# Scatter plot: raw data before clustering
plt.figure(figsize=(7, 5))
plt.scatter(df['AnnualIncome'], df['SpendingScore'],
            alpha=0.6, color='slategray', edgecolors='white', s=60)
plt.title("Annual Income vs Spending Score (Before Clustering)", fontweight='bold')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1–100)")
plt.tight_layout()
plt.savefig("scatter_before_clustering.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Pre-clustering scatter saved")

# Correlation heatmap
plt.figure(figsize=(5, 4))
corr = df[['Age','AnnualIncome','SpendingScore']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Correlation heatmap saved")

# ── SECTION 5: Feature Selection & Scaling ───────────────────
X = df[['AnnualIncome', 'SpendingScore']].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n── Selected Features ──")
print("  • Annual Income (k$)")
print("  • Spending Score (1–100)")
print(f"\nScaled data shape: {X_scaled.shape}")
print("Sample scaled values (first 5 rows):")
print(pd.DataFrame(X_scaled, columns=['AnnualIncome_scaled','SpendingScore_scaled']).head())

# ── SECTION 6: Elbow Method ───────────────────────────────────
inertia = []
k_range = range(1, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(k_range, inertia, marker='o', color='royalblue', linewidth=2, markersize=8)
plt.axvline(x=3, color='red', linestyle='--', linewidth=1.5, label='Chosen K=3')
plt.title("Elbow Method — Optimal Number of Clusters", fontweight='bold')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia (Within-Cluster Sum of Squares)")
plt.xticks(k_range)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("elbow_curve.png", dpi=150, bbox_inches='tight')
plt.show()

print("\n── Inertia Values ──")
for k, val in zip(k_range, inertia):
    print(f"  K={k}: {val:.2f}")
print("\n✅ Elbow curve saved as elbow_curve.png")

# ── SECTION 7: K-Means Clustering (K=3) ──────────────────────
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

print("\n── Cluster Labels Assigned ──")
print(df[['CustomerID','AnnualIncome','SpendingScore','Cluster']].head(15).to_string(index=False))
print(f"\nCluster distribution:\n{df['Cluster'].value_counts().sort_index()}")

# ── SECTION 8: Silhouette Score ───────────────────────────────
sil_score = silhouette_score(X_scaled, df['Cluster'])
print(f"\n── Silhouette Score ──")
print(f"  Score: {sil_score:.4f}")
print("  Interpretation:")
print("  • 0.51–0.70 = Reasonable structure")
print("  • 0.71–1.00 = Strong structure")
if sil_score > 0.5:
    print(f"  ✅ Score of {sil_score:.2f} indicates well-separated clusters.")

# ── SECTION 9: Cluster Visualization ─────────────────────────
cluster_colors  = {0: '#2196F3', 1: '#4CAF50', 2: '#FF5722'}
cluster_labels  = {0: 'Cluster 0', 1: 'Cluster 1', 2: 'Cluster 2'}

plt.figure(figsize=(9, 6))
for c in [0, 1, 2]:
    subset = df[df['Cluster'] == c]
    plt.scatter(subset['AnnualIncome'], subset['SpendingScore'],
                label=cluster_labels[c], color=cluster_colors[c],
                alpha=0.7, edgecolors='white', s=70)

# Plot centroids (inverse transform back to original scale)
centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centroids_orig[:, 0], centroids_orig[:, 1],
            c='black', marker='X', s=220, zorder=5, label='Centroids')

plt.title("K-Means Clustering (K=3) — Annual Income vs Spending Score",
          fontweight='bold')
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1–100)")
plt.legend()
plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig("kmeans_clusters.png", dpi=150, bbox_inches='tight')
plt.show()
print("✅ Cluster scatter plot saved as kmeans_clusters.png")

# ── SECTION 10: Cluster Interpretation ───────────────────────
cluster_summary = df.groupby('Cluster')[['AnnualIncome','SpendingScore','Age']].mean().round(1)
cluster_summary['Count'] = df.groupby('Cluster').size()
print("\n── Cluster Summary Table ──")
print(cluster_summary.to_string())

# Map cluster numbers to business labels based on centroids
# Identify which cluster is which by centroid values
c_income    = centroids_orig[:, 0]
c_spending  = centroids_orig[:, 1]

for i in range(3):
    label = ""
    if c_income[i] > 65 and c_spending[i] < 40:
        label = "HIGH INCOME – LOW SPENDING (Cautious Savers)"
    elif c_income[i] < 50 and c_spending[i] > 55:
        label = "LOW INCOME – HIGH SPENDING (Impulsive Shoppers)"
    else:
        label = "MID INCOME – MODERATE SPENDING (Balanced Shoppers)"
    print(f"  Cluster {i}: Income={c_income[i]:.0f}k$, Spending={c_spending[i]:.0f} → {label}")

print("\n✅ All sections complete. Output above.")




