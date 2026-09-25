import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from pathlib import Path
from datetime import datetime

sns.set_theme(style='whitegrid', palette='tab20')

# ── Data ──────────────────────────────────────────────────────────────────────
dir = Path(__file__).parent

def save(fig, name):
    fig.savefig(dir / name, dpi=150, bbox_inches='tight')
    plt.show()

A3_data = pd.read_csv(dir / 'A3 set.txt', sep=r'\s+', header=None)
A3_ga_cb_data = pd.read_table(dir / 'a3-ga-cb.txt')
A3_ga_pa_data = pd.read_table(dir / 'a3-ga.pa')

seed = 42
np.random.seed(seed)
K = 48

xs = A3_data.iloc[:, 0]
ys = A3_data.iloc[:, 1]

# ── KMeans ────────────────────────────────────────────────────────────────────
model_kmeans     = KMeans(n_clusters=K, init='random', random_state=seed, max_iter=300)
labels_kmeans    = model_kmeans.fit_predict(A3_data)
centroids_kmeans = model_kmeans.cluster_centers_

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=xs, y=ys, hue=labels_kmeans, palette='tab20', alpha=0.5,
                s=15, legend=False, ax=ax)
ax.scatter(centroids_kmeans[:, 0], centroids_kmeans[:, 1],
           c='red', marker='x', s=80, linewidths=1.5, zorder=5, label='Centroids')
ax.set_title(f'KMeans Clustering (k={K})')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()
plt.tight_layout()
save(fig, '01_kmeans_clusters.png')

# ── GMM ───────────────────────────────────────────────────────────────────────
def draw_gmm_ellipses(ax, gmm, n_std=2.0):
    """Overlay covariance ellipses for every GMM component."""
    for mean, cov in zip(gmm.means_, gmm.covariances_):
        cov_type = gmm.covariance_type
        if cov_type in ('full', 'tied'):
            cov_2d = cov[:2, :2]
        elif cov_type == 'diag':
            cov_2d = np.diag(cov[:2])
        else:  # spherical
            cov_2d = np.eye(2) * cov

        eigvals, eigvecs = np.linalg.eigh(cov_2d)
        order            = eigvals.argsort()[::-1]
        eigvals, eigvecs = eigvals[order], eigvecs[:, order]
        angle            = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
        width, height    = 2 * n_std * np.sqrt(np.abs(eigvals))
        ellipse = Ellipse(
            xy=(mean[0], mean[1]), width=width, height=height, angle=angle,
            edgecolor='k', facecolor='none', linewidth=0.6, alpha=0.5,
        )
        ax.add_patch(ellipse)

def compute_td2(X, labels):
    """Total Distance Squared: sum of squared Euclidean distances to cluster centroids."""
    X   = np.asarray(X)
    td2 = 0.0
    for lbl in np.unique(labels):
        pts      = X[labels == lbl]
        centroid = pts.mean(axis=0)
        td2     += np.sum((pts - centroid) ** 2)
    return td2

model_gmm = GaussianMixture(n_components=K, random_state=seed, max_iter=300)
model_gmm.fit(A3_data)
labels_gmm    = model_gmm.predict(A3_data)
centroids_gmm = model_gmm.means_

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=xs, y=ys, hue=labels_gmm, palette='tab20', alpha=0.5,
                s=15, legend=False, ax=ax)
ax.scatter(centroids_gmm[:, 0], centroids_gmm[:, 1],
           c='blue', marker='x', s=80, linewidths=1.5, zorder=5, label='Means')
draw_gmm_ellipses(ax, model_gmm)
ax.set_title(f'GMM Clustering (k={K}) with covariance ellipses')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.legend()
plt.tight_layout()
save(fig, '02_gmm_clusters.png')

# ── Agglomerative Clustering ──────────────────────────────────────────────────
model_agg  = AgglomerativeClustering(n_clusters=K)
labels_agg = model_agg.fit_predict(A3_data)

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=xs, y=ys, hue=labels_agg, palette='tab20', alpha=0.5,
                s=15, legend=False, ax=ax)
ax.set_title(f'Agglomerative Clustering (k={K})')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
plt.tight_layout()
save(fig, '03_agglomerative_clusters.png')

# Dendrogram on a random sample to keep it readable
sample_idx = np.random.choice(len(A3_data), size=min(300, len(A3_data)), replace=False)
mergings   = linkage(A3_data.iloc[sample_idx], method='complete')
fig, ax    = plt.subplots(figsize=(12, 5))
dendrogram(mergings, leaf_rotation=90, leaf_font_size=4, ax=ax,
           color_threshold=0.7 * max(mergings[:, 2]))
ax.set_title('Hierarchical Clustering Dendrogram (300-point sample)')
ax.set_xlabel('Sample index')
ax.set_ylabel('Distance')
sns.despine(ax=ax)
plt.tight_layout()
save(fig, '04_dendrogram.png')

# ── Distribution Visualisation ────────────────────────────────────────────────
# Feature distributions for the 5 largest GMM components
top_components = np.argsort(model_gmm.weights_)[::-1][:5]
dist_df = pd.DataFrame({
    'Feature 1': xs.values,
    'Feature 2': ys.values,
    'Component': [f'Comp {c}' for c in labels_gmm],
})
dist_df = dist_df[dist_df['Component'].isin([f'Comp {c}' for c in top_components])]

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(data=dist_df, x='Feature 1', hue='Component', bins=30,
             kde=True, alpha=0.45, ax=axes[0])
sns.histplot(data=dist_df, x='Feature 2', hue='Component', bins=30,
             kde=True, alpha=0.45, ax=axes[1])
axes[0].set_title('Feature 1 distribution by GMM component (top 5)')
axes[1].set_title('Feature 2 distribution by GMM component (top 5)')
plt.tight_layout()
save(fig, '05_gmm_distributions.png')

# ── Metrics ───────────────────────────────────────────────────────────────────
sil_kmeans = silhouette_score(A3_data, labels_kmeans)
sil_gmm    = silhouette_score(A3_data, labels_gmm)
sil_agg    = silhouette_score(A3_data, labels_agg)
print(f"Silhouette Score  — KMeans:        {sil_kmeans:.4f}")
print(f"Silhouette Score  — GMM:           {sil_gmm:.4f}")
print(f"Silhouette Score  — Agglomerative: {sil_agg:.4f}")
print(f"KMeans Inertia:   {model_kmeans.inertia_:.2f}")
print(f"GMM BIC:          {model_gmm.bic(A3_data):.2f}")

# Silhouette bar chart comparing all three models
sil_df = pd.DataFrame({
    'Model':            ['KMeans', 'GMM', 'Agglomerative'],
    'Silhouette Score': [sil_kmeans, sil_gmm, sil_agg],
})
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=sil_df, x='Model', y='Silhouette Score',
            palette='muted', ax=ax)
ax.set_ylim(0, 1)
ax.set_title('Silhouette Score Comparison')
for bar, val in zip(ax.patches, sil_df['Silhouette Score']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
            f'{val:.4f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
save(fig, '06_silhouette_comparison.png')

# ── TD² Statistic ─────────────────────────────────────────────────────────────
# KMeans inertia == TD² by definition; compute explicitly for GMM and Agglomerative
td2_kmeans = model_kmeans.inertia_
td2_gmm    = compute_td2(A3_data, labels_gmm)
td2_agg    = compute_td2(A3_data, labels_agg)
print(f"\nTD²  — KMeans:        {td2_kmeans:.2f}")
print(f"TD²  — GMM:           {td2_gmm:.2f}")
print(f"TD²  — Agglomerative: {td2_agg:.2f}")

td2_df = pd.DataFrame({
    'Model': ['KMeans', 'GMM', 'Agglomerative'],
    'TD²':   [td2_kmeans, td2_gmm, td2_agg],
})
fig, ax = plt.subplots(figsize=(6, 4))
sns.barplot(data=td2_df, x='Model', y='TD²', palette='muted', ax=ax)
ax.set_title(f'TD² Statistic Comparison (k={K})')
for bar, val in zip(ax.patches, td2_df['TD²']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
            f'{val:,.0f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
save(fig, '08_td2_comparison.png')

# ── Elbow curve & Silhouette sweep ────────────────────────────────────────────
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(A3_data)
k_range  = list(range(2, 51))
inertias, sil_scores = [], []

for k in k_range:
    km  = KMeans(n_clusters=k, random_state=seed)
    lbl = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, lbl))
    print(f"k={k:2d}  inertia={km.inertia_:10.1f}  silhouette={sil_scores[-1]:.4f}  TD²={km.inertia_:10.1f}")

sweep_df = pd.DataFrame({'k': k_range, 'Inertia': inertias, 'Silhouette Score': sil_scores})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
sns.lineplot(data=sweep_df, x='k', y='Inertia', marker='o', ax=ax1)
ax1.set_title('Elbow Curve')
ax1.set_xlabel('Number of clusters (k)')
ax1.set_xticks(k_range[::2])

sns.lineplot(data=sweep_df, x='k', y='Silhouette Score', marker='o',
             color='darkorange', ax=ax2)
ax2.set_title('Silhouette Score vs k')
ax2.set_xlabel('Number of clusters (k)')
ax2.set_xticks(k_range[::2])

plt.tight_layout()
save(fig, '07_elbow_silhouette_sweep.png')

# ── TD² Sweep & Rate of Change ────────────────────────────────────────────────
# inertias == TD² for KMeans on scaled data; delta shows where adding a cluster
# stops providing meaningful compactness gains (ratio levels off near 1.0)
delta_td2    = [inertias[i] / inertias[i - 1] for i in range(1, len(inertias))]
td2_sweep_df = pd.DataFrame({'k': k_range,      'TD²':          inertias})
delta_df     = pd.DataFrame({'k': k_range[1:],  'TD² ratio (k/k-1)': delta_td2})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

sns.lineplot(data=td2_sweep_df, x='k', y='TD²', marker='o', ax=ax1)
ax1.set_title('TD² vs k')
ax1.set_xlabel('Number of clusters (k)')
ax1.set_xticks(k_range)

sns.lineplot(data=delta_df, x='k', y='TD² ratio (k/k-1)', marker='o',
             color='steelblue', ax=ax2)
ax2.axhline(y=0.9, color='red', linestyle='--', linewidth=1, label='0.9 threshold')
ax2.set_title('TD² Rate of Change  (TD²(k) / TD²(k−1))')
ax2.set_xlabel('Number of clusters (k)')
ax2.set_xticks(k_range[1:])
ax2.legend()

plt.tight_layout()
save(fig, '09_td2_sweep.png')

# ── KMeans: Stability across random states ────────────────────────────────────
N_STATES = 10
ari_means, ari_stds = [], []

print("\nKMeans stability sweep...")
for k in k_range:
    runs = [KMeans(n_clusters=k, random_state=rs).fit_predict(X_scaled)
            for rs in range(N_STATES)]
    aris = [adjusted_rand_score(runs[i], runs[j])
            for i in range(N_STATES) for j in range(i + 1, N_STATES)]
    ari_means.append(np.mean(aris))
    ari_stds.append(np.std(aris))
    print(f"  k={k:2d}  mean_ARI={ari_means[-1]:.4f}  std={ari_stds[-1]:.4f}")

stability_df = pd.DataFrame({
    'k':        k_range,
    'Mean ARI': ari_means,
    'Std ARI':  ari_stds,
})
fig, ax = plt.subplots(figsize=(10, 4))
sns.lineplot(data=stability_df, x='k', y='Mean ARI', marker='o', ax=ax)
ax.fill_between(
    k_range,
    np.array(ari_means) - np.array(ari_stds),
    np.array(ari_means) + np.array(ari_stds),
    alpha=0.2, label='±1 std',
)
ax.set_title(f'KMeans Stability: Mean Pairwise ARI across {N_STATES} random states')
ax.set_xlabel('Number of clusters (k)')
ax.set_ylabel('Adjusted Rand Index')
ax.set_xticks(k_range)
ax.legend()
plt.tight_layout()
save(fig, '10_kmeans_stability_ari.png')

# ── GMM: Covariance-type × n_components sweep ─────────────────────────────────
COV_TYPES    = ['full', 'tied', 'diag', 'spherical']
N_COMP_RANGE = list(range(max(2, K - 5), K + 6))
GMM_STATES   = 5

gmm_sweep_rows = []
print("\nGMM covariance × n_components sweep...")
for cov_type in COV_TYPES:
    for n_comp in N_COMP_RANGE:
        gmm_s = GaussianMixture(n_components=n_comp, covariance_type=cov_type,
                                random_state=seed, max_iter=300)
        gmm_s.fit(A3_data)
        lbl = gmm_s.predict(A3_data)
        sil = silhouette_score(A3_data, lbl) if len(np.unique(lbl)) > 1 else np.nan
        gmm_sweep_rows.append({
            'covariance_type': cov_type,
            'n_components':    n_comp,
            'log_likelihood':  gmm_s.score(A3_data) * len(A3_data),
            'BIC':             gmm_s.bic(A3_data),
            'AIC':             gmm_s.aic(A3_data),
            'silhouette':      sil,
        })
        print(f"  {cov_type:10s}  n={n_comp}  "
              f"BIC={gmm_sweep_rows[-1]['BIC']:.1f}  sil={sil:.4f}")

gmm_sweep_df = pd.DataFrame(gmm_sweep_rows)

fig, axes = plt.subplots(2, 2, figsize=(14, 9))
metrics = ['BIC', 'AIC', 'log_likelihood', 'silhouette']
titles  = ['BIC (lower = better)', 'AIC (lower = better)',
           'Log-Likelihood (higher = better)', 'Silhouette Score (higher = better)']
for ax, metric, title in zip(axes.flat, metrics, titles):
    sns.lineplot(data=gmm_sweep_df, x='n_components', y=metric,
                 hue='covariance_type', marker='o', ax=ax)
    ax.axvline(x=K, color='red', linestyle='--', linewidth=1,
               alpha=0.6, label=f'K={K}')
    ax.set_title(title)
    ax.set_xlabel('n_components')
    ax.legend(fontsize=7)
plt.suptitle('GMM: Covariance Type × n_components Sweep', y=1.01)
plt.tight_layout()
save(fig, '11_gmm_cov_sweep.png')

# GMM stability: pairwise ARI across random states per covariance type at K
gmm_ari_rows = []
print(f"\nGMM stability sweep (n_components={K})...")
for cov_type in COV_TYPES:
    runs = []
    for rs in range(GMM_STATES):
        gmm_s = GaussianMixture(n_components=K, covariance_type=cov_type,
                                random_state=rs, max_iter=300)
        gmm_s.fit(A3_data)
        runs.append(gmm_s.predict(A3_data))
    aris = [adjusted_rand_score(runs[i], runs[j])
            for i in range(GMM_STATES) for j in range(i + 1, GMM_STATES)]
    gmm_ari_rows.append({
        'covariance_type': cov_type,
        'mean_ARI':        np.mean(aris),
        'std_ARI':         np.std(aris),
    })
    print(f"  {cov_type:10s}  mean_ARI={gmm_ari_rows[-1]['mean_ARI']:.4f}  "
          f"std={gmm_ari_rows[-1]['std_ARI']:.4f}")

gmm_ari_df = pd.DataFrame(gmm_ari_rows)
fig, ax = plt.subplots(figsize=(7, 4))
sns.barplot(data=gmm_ari_df, x='covariance_type', y='mean_ARI', palette='muted', ax=ax)
ax.errorbar(
    x=range(len(COV_TYPES)), y=gmm_ari_df['mean_ARI'],
    yerr=gmm_ari_df['std_ARI'], fmt='none', color='black', capsize=5,
)
ax.set_ylim(0, 1)
ax.set_title(f'GMM Stability: Mean Pairwise ARI across {GMM_STATES} random states  (k={K})')
ax.set_xlabel('Covariance Type')
ax.set_ylabel('Adjusted Rand Index')
for bar, val in zip(ax.patches, gmm_ari_df['mean_ARI']):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.03,
            f'{val:.3f}', ha='center', va='bottom', fontsize=9)
plt.tight_layout()
save(fig, '12_gmm_stability_ari.png')

# ── Agglomerative: Linkage sweep ──────────────────────────────────────────────
LINKAGE_METHODS = ['single', 'complete', 'average', 'ward']
agg_metric_rows = []
agg_labels_dict = {}

print("\nAgglomerative linkage sweep...")
for link in LINKAGE_METHODS:
    agg_s = AgglomerativeClustering(n_clusters=K, linkage=link)
    lbl   = agg_s.fit_predict(A3_data)
    sil   = silhouette_score(A3_data, lbl)
    td2   = compute_td2(A3_data, lbl)
    agg_labels_dict[link] = lbl
    agg_metric_rows.append({'linkage': link, 'silhouette': sil, 'TD²': td2})
    print(f"  {link:10s}  silhouette={sil:.4f}  TD²={td2:,.2f}")

# 2×2 scatter grid showing cluster shapes per linkage method
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
for ax, link in zip(axes.flat, LINKAGE_METHODS):
    sns.scatterplot(x=xs, y=ys, hue=agg_labels_dict[link], palette='tab20',
                    alpha=0.5, s=10, legend=False, ax=ax)
    ax.set_title(f'{link} linkage')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
plt.suptitle(f'Agglomerative Clustering Shapes by Linkage Method (k={K})')
plt.tight_layout()
save(fig, '13_agg_linkage_scatter.png')

# Silhouette + TD² bar charts per linkage
agg_metrics_df = pd.DataFrame(agg_metric_rows)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

sns.barplot(data=agg_metrics_df, x='linkage', y='silhouette', palette='muted', ax=ax1)
ax1.set_ylim(0, 1)
ax1.set_title('Agglomerative: Silhouette Score by Linkage')
for bar, val in zip(ax1.patches, agg_metrics_df['silhouette']):
    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{val:.4f}', ha='center', va='bottom', fontsize=9)

sns.barplot(data=agg_metrics_df, x='linkage', y='TD²', palette='muted', ax=ax2)
ax2.set_title('Agglomerative: TD² by Linkage')
for bar, val in zip(ax2.patches, agg_metrics_df['TD²']):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
             f'{val:,.0f}', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
save(fig, '14_agg_linkage_metrics.png')

print(f"\nAll plots saved to: {dir}")