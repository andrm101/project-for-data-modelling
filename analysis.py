"""
A3 Dataset Clustering Analysis
================================
Dataset : A3 benchmark – 7500 points, 2D, 50 Gaussian clusters
Algorithms:
  1. K-Means  (Brute-Force: random init, 50 restarts)
  2. K-Means+ (K-Means++ init, 20 restarts)
  3. GMM/EM   (Gaussian Mixture Model – soft / probabilistic)
  4. GA-VQ    (Genetic Algorithm VQ – reference from provided files)
Metrics  : TD2, Log-Likelihood, Silhouette, Davies-Bouldin,
           Calinski-Harabasz, BIC/AIC, Instability
"""

import sys, os, time, warnings
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (silhouette_score,
                              davies_bouldin_score,
                              calinski_harabasz_score)
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────
K            = 50           # true number of clusters in A3
SEED         = 42
N_INIT_BF    = 50           # random restarts for brute-force K-Means
N_INIT_PP    = 20           # restarts for K-Means++
N_INIT_GMM   = 5            # EM restarts for GMM
STAB_RUNS    = 10           # runs for instability measurement
ELBOW_RANGE  = range(10, 81, 5)   # K values for elbow curve
SIL_SAMPLE   = 2000         # subsample size for silhouette (O(n²))

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
OUT_DIR   = os.path.join(BASE_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

CMAP = plt.get_cmap("tab20", K)   # 50-colour map

# ──────────────────────────────────────────────────────────────
# 1. DATA LOADING
# ──────────────────────────────────────────────────────────────
def load_xy(path):
    """Load files of the form: x  y  (or optionally: index  x  y)."""
    pts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            cols = line.split()
            if len(cols) >= 2:
                try:
                    # Works for both "x y" and "idx x y" layouts
                    pts.append([float(cols[-2]), float(cols[-1])])
                except ValueError:
                    pass
    return np.array(pts, dtype=np.float64)

def load_partition(path):
    """Load VQ .pa file – skip header block ending with '---...'."""
    labels, past_header = [], False
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if "---" in s:
                past_header = True
                continue
            if past_header and s:
                try:
                    labels.append(int(s))
                except ValueError:
                    pass
    return np.array(labels, dtype=np.int32)

print("Loading data …")
X         = load_xy(os.path.join(BASE_DIR, "A3 set.txt"))
cb_ga     = load_xy(os.path.join(BASE_DIR, "a3-ga-cb.txt"))
labels_ga_raw = load_partition(os.path.join(BASE_DIR, "a3-ga.pa"))

# Remap GA labels 1-50 → 0-49 for sklearn consistency
unique_ga = np.unique(labels_ga_raw)
ga_map    = {v: i for i, v in enumerate(sorted(unique_ga))}
labels_ga = np.array([ga_map[l] for l in labels_ga_raw[:len(X)]])

print(f"  Dataset : {X.shape[0]} points | X=[{X[:,0].min():.0f},{X[:,0].max():.0f}] "
      f"Y=[{X[:,1].min():.0f},{X[:,1].max():.0f}]")
print(f"  Codebook: {cb_ga.shape[0]} centroids")
print(f"  GA-VQ   : {len(labels_ga)} labels | {len(np.unique(labels_ga))} unique clusters")

# ──────────────────────────────────────────────────────────────
# 2. METRIC HELPERS
# ──────────────────────────────────────────────────────────────
def td2(X, centroids, labels):
    """Total Distortion squared = Σ ||xᵢ − cₖ(ᵢ)||²  (=WCSS/inertia)."""
    total = 0.0
    for k in np.unique(labels):
        mask = labels == k
        diff = X[mask] - centroids[k]
        total += float(np.einsum("ij,ij->", diff, diff))
    return total

def silhouette_sampled(X, labels, n=SIL_SAMPLE, seed=SEED):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(n, len(X)), replace=False)
    return silhouette_score(X[idx], labels[idx])

# ──────────────────────────────────────────────────────────────
# 3. GA-VQ REFERENCE METRICS
# ──────────────────────────────────────────────────────────────
print("\n[GA-VQ] Computing reference metrics …")
td2_ga  = td2(X, cb_ga, labels_ga)
sil_ga  = silhouette_sampled(X, labels_ga)
db_ga   = davies_bouldin_score(X, labels_ga)
ch_ga   = calinski_harabasz_score(X, labels_ga)
print(f"  TD2={td2_ga:.4e}  Sil={sil_ga:.4f}  DB={db_ga:.4f}  CH={ch_ga:.2f}")

# ──────────────────────────────────────────────────────────────
# 4. K-MEANS  BRUTE-FORCE  (random init, many restarts)
# ──────────────────────────────────────────────────────────────
print(f"\n[K-Means BF] random init, {N_INIT_BF} restarts …")
t0 = time.perf_counter()
km_bf = KMeans(n_clusters=K, init="random", n_init=N_INIT_BF,
               max_iter=300, random_state=SEED)
km_bf.fit(X)
t_bf = time.perf_counter() - t0
labels_bf = km_bf.labels_
centroids_bf = km_bf.cluster_centers_
td2_bf  = float(km_bf.inertia_)
sil_bf  = silhouette_sampled(X, labels_bf)
db_bf   = davies_bouldin_score(X, labels_bf)
ch_bf   = calinski_harabasz_score(X, labels_bf)
print(f"  TD2={td2_bf:.4e}  Sil={sil_bf:.4f}  DB={db_bf:.4f}  CH={ch_bf:.2f}  t={t_bf:.1f}s")

# ──────────────────────────────────────────────────────────────
# 5. K-MEANS++ (smart initialization)
# ──────────────────────────────────────────────────────────────
print(f"\n[K-Means++] k-means++ init, {N_INIT_PP} restarts …")
t0 = time.perf_counter()
km_pp = KMeans(n_clusters=K, init="k-means++", n_init=N_INIT_PP,
               max_iter=300, random_state=SEED)
km_pp.fit(X)
t_pp = time.perf_counter() - t0
labels_pp = km_pp.labels_
centroids_pp = km_pp.cluster_centers_
td2_pp  = float(km_pp.inertia_)
sil_pp  = silhouette_sampled(X, labels_pp)
db_pp   = davies_bouldin_score(X, labels_pp)
ch_pp   = calinski_harabasz_score(X, labels_pp)
print(f"  TD2={td2_pp:.4e}  Sil={sil_pp:.4f}  DB={db_pp:.4f}  CH={ch_pp:.2f}  t={t_pp:.1f}s")

# ──────────────────────────────────────────────────────────────
# 6. GMM / EM  (soft / probabilistic clustering)
# ──────────────────────────────────────────────────────────────
print(f"\n[GMM/EM] full-covariance, {N_INIT_GMM} restarts …")
t0 = time.perf_counter()
gmm = GaussianMixture(n_components=K, covariance_type="full",
                      n_init=N_INIT_GMM, max_iter=300, random_state=SEED)
gmm.fit(X)
t_gmm  = time.perf_counter() - t0
labels_gmm    = gmm.predict(X)
resp_gmm      = gmm.predict_proba(X)          # soft memberships (N×50)
ll_total      = gmm.score(X) * len(X)         # total log-likelihood
bic_gmm       = gmm.bic(X)
aic_gmm       = gmm.aic(X)
td2_gmm       = td2(X, gmm.means_, labels_gmm)
sil_gmm       = silhouette_sampled(X, labels_gmm)
db_gmm        = davies_bouldin_score(X, labels_gmm)
ch_gmm        = calinski_harabasz_score(X, labels_gmm)
print(f"  TD2={td2_gmm:.4e}  LL={ll_total:.4e}  BIC={bic_gmm:.4e}  AIC={aic_gmm:.4e}")
print(f"  Sil={sil_gmm:.4f}  DB={db_gmm:.4f}  CH={ch_gmm:.2f}  t={t_gmm:.1f}s")

# ──────────────────────────────────────────────────────────────
# 7. INSTABILITY ANALYSIS (variance of TD2 across seeds)
# ──────────────────────────────────────────────────────────────
print(f"\n[Instability] {STAB_RUNS} runs per algorithm …")
td2_runs_bf, td2_runs_pp, td2_runs_gmm = [], [], []
for run in range(STAB_RUNS):
    s = run * 7
    td2_runs_bf.append(KMeans(K, init="random", n_init=1, max_iter=200,
                               random_state=s).fit(X).inertia_)
    td2_runs_pp.append(KMeans(K, init="k-means++", n_init=1, max_iter=200,
                               random_state=s).fit(X).inertia_)
    g = GaussianMixture(K, covariance_type="full", n_init=1, max_iter=200,
                        random_state=s).fit(X)
    td2_runs_gmm.append(td2(X, g.means_, g.predict(X)))

instab_bf  = np.std(td2_runs_bf)
instab_pp  = np.std(td2_runs_pp)
instab_gmm = np.std(td2_runs_gmm)
print(f"  K-Means BF  σ(TD2) = {instab_bf:.4e}")
print(f"  K-Means++   σ(TD2) = {instab_pp:.4e}")
print(f"  GMM/EM      σ(TD2) = {instab_gmm:.4e}")

# ──────────────────────────────────────────────────────────────
# 8. ELBOW CURVE  (TD2 vs K for K-Means++)
# ──────────────────────────────────────────────────────────────
print("\n[Elbow] running K-Means++ for K =", list(ELBOW_RANGE), "…")
elbow_k, elbow_td2 = [], []
for k in ELBOW_RANGE:
    km = KMeans(n_clusters=k, init="k-means++", n_init=5, max_iter=200,
                random_state=SEED)
    km.fit(X)
    elbow_k.append(k)
    elbow_td2.append(km.inertia_)
    print(f"  K={k:3d}  TD2={km.inertia_:.4e}")

# ──────────────────────────────────────────────────────────────
# 9. SUMMARY TABLE
# ──────────────────────────────────────────────────────────────
summary = pd.DataFrame({
    "Algorithm"      : ["K-Means BF", "K-Means++", "GMM/EM", "GA-VQ (ref)"],
    "TD2"            : [td2_bf,  td2_pp,  td2_gmm,  td2_ga],
    "Log-Likelihood" : [None,    None,    ll_total, None],
    "BIC"            : [None,    None,    bic_gmm,  None],
    "AIC"            : [None,    None,    aic_gmm,  None],
    "Silhouette ↑"   : [sil_bf,  sil_pp,  sil_gmm,  sil_ga],
    "Davies-Bouldin ↓": [db_bf,  db_pp,   db_gmm,   db_ga],
    "Calinski-H ↑"   : [ch_bf,   ch_pp,   ch_gmm,   ch_ga],
    "σ(TD2) instab." : [instab_bf, instab_pp, instab_gmm, None],
    "Time (s)"       : [t_bf,    t_pp,    t_gmm,    None],
})
print("\n" + "="*90)
print("FULL METRICS SUMMARY")
print("="*90)
print(summary.to_string(index=False))
summary.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)

# ──────────────────────────────────────────────────────────────
# 10. VISUALIZATIONS
# ──────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams.update({"figure.dpi": 150, "savefig.bbox": "tight"})

ALGO_CONFIGS = [
    ("K-Means BF\n(random init, 50 restarts)",  labels_bf,  centroids_bf,  td2_bf,  sil_bf),
    ("K-Means++\n(smart init, 20 restarts)",     labels_pp,  centroids_pp,  td2_pp,  sil_pp),
    ("GMM / EM\n(soft, full covariance)",        labels_gmm, gmm.means_,    td2_gmm, sil_gmm),
    ("GA-VQ Reference\n(Genetic Algorithm)",     labels_ga,  cb_ga,         td2_ga,  sil_ga),
]

# ── Fig 1 : Raw data ──────────────────────────────────────────
print("\nSaving figures …")
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(X[:, 0], X[:, 1], s=2, alpha=0.4, color="steelblue", linewidths=0)
ax.set_title("A3 Benchmark Dataset  (7 500 points, 2-D)", fontsize=13, fontweight="bold")
ax.set_xlabel("x"); ax.set_ylabel("y")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig1_raw_data.png"))
plt.close(fig)

# ── Fig 2-5 : Cluster assignments per algorithm ───────────────
for idx, (title, lbl, cents, td2_val, sil_val) in enumerate(ALGO_CONFIGS, start=2):
    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(X[:, 0], X[:, 1], c=lbl, cmap="tab20", s=3,
                    alpha=0.5, linewidths=0, vmin=0, vmax=K - 1)
    ax.scatter(cents[:, 0], cents[:, 1], c="white", edgecolors="black",
               s=120, zorder=5, linewidths=1.5, marker="*")
    ax.set_title(f"{title}\nTD² = {td2_val:.3e}   Silhouette = {sil_val:.4f}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(sc, ax=ax, label="Cluster ID", shrink=0.8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"fig{idx}_cluster_{idx-1}.png"))
    plt.close(fig)

# ── Fig 6 : GMM soft-assignment heatmap (max responsibility) ──
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
# Left: hard assignment
axes[0].scatter(X[:, 0], X[:, 1], c=labels_gmm, cmap="tab20", s=3,
                alpha=0.5, linewidths=0)
axes[0].set_title("GMM – Hard Assignment (argmax)", fontweight="bold")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
# Right: maximum responsibility (confidence)
max_resp = resp_gmm.max(axis=1)
sc2 = axes[1].scatter(X[:, 0], X[:, 1], c=max_resp, cmap="RdYlGn",
                       s=3, alpha=0.6, linewidths=0, vmin=0, vmax=1)
axes[1].set_title("GMM – Max Responsibility\n(green = high confidence)", fontweight="bold")
axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
fig.colorbar(sc2, ax=axes[1], label="P(best cluster | xᵢ)", shrink=0.8)
fig.suptitle("Gaussian Mixture Model – Soft Clustering", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig6_gmm_soft.png"))
plt.close(fig)

# ── Fig 7 : TD2 comparison bar chart ─────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))
algos  = ["K-Means\nBrute Force", "K-Means++", "GMM/EM", "GA-VQ\n(Reference)"]
td2s   = [td2_bf, td2_pp, td2_gmm, td2_ga]
colors = ["#E63946", "#F4A261", "#2A9D8F", "#457B9D"]
bars = ax.bar(algos, td2s, color=colors, edgecolor="black", width=0.55)
for bar, val in zip(bars, td2s):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.005,
            f"{val:.3e}", ha="center", va="bottom", fontsize=9, fontweight="bold")
ax.set_ylabel("Total Distortion Squared (TD²)", fontsize=11)
ax.set_title("TD² Comparison  –  Lower is Better", fontsize=13, fontweight="bold")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig7_td2_comparison.png"))
plt.close(fig)

# ── Fig 8 : Silhouette + Davies-Bouldin + Calinski-Harabasz ──
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metric_data = {
    "Silhouette ↑": ([sil_bf, sil_pp, sil_gmm, sil_ga], True),
    "Davies-Bouldin ↓": ([db_bf, db_pp, db_gmm, db_ga], False),
    "Calinski-Harabasz ↑": ([ch_bf, ch_pp, ch_gmm, ch_ga], True),
}
labels_short = ["K-Means BF", "K-Means++", "GMM/EM", "GA-VQ"]
for ax, (metric, (vals, higher_better)) in zip(axes, metric_data.items()):
    bars = ax.bar(labels_short, vals, color=colors, edgecolor="black", width=0.55)
    best_idx = int(np.argmax(vals) if higher_better else np.argmin(vals))
    bars[best_idx].set_edgecolor("gold")
    bars[best_idx].set_linewidth(3)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f"{v:.3f}" if abs(v) < 1000 else f"{v:.0f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_title(metric, fontsize=11, fontweight="bold")
    ax.set_xticklabels(labels_short, rotation=15, ha="right", fontsize=9)
    ax.set_ylabel("")
fig.suptitle("Clustering Quality Metrics Comparison", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig8_quality_metrics.png"))
plt.close(fig)

# ── Fig 9 : Instability (TD2 distribution per algorithm) ─────
fig, ax = plt.subplots(figsize=(9, 5))
data_runs = [td2_runs_bf, td2_runs_pp, td2_runs_gmm]
bp = ax.boxplot(data_runs, labels=["K-Means\nBrute Force", "K-Means++", "GMM/EM"],
                patch_artist=True, widths=0.5,
                boxprops=dict(facecolor="lightsteelblue", linewidth=1.5),
                medianprops=dict(color="crimson", linewidth=2))
for i, (vals, col) in enumerate(zip(data_runs, colors[:3]), start=1):
    x = np.random.normal(i, 0.05, size=len(vals))
    ax.scatter(x, vals, s=40, zorder=5, color=col, edgecolors="black", linewidths=0.8)
ax.set_ylabel("TD² per single run", fontsize=11)
ax.set_title(f"Algorithm Instability  –  TD² variance across {STAB_RUNS} runs\n"
             f"(σ: BF={instab_bf:.3e}, K++={instab_pp:.3e}, GMM={instab_gmm:.3e})",
             fontsize=11, fontweight="bold")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig9_instability.png"))
plt.close(fig)

# ── Fig 10 : Elbow curve ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(elbow_k, elbow_td2, "o-", color="#457B9D", linewidth=2, markersize=7,
        markerfacecolor="white", markeredgewidth=2)
ax.axvline(K, color="crimson", linestyle="--", linewidth=1.5,
           label=f"True K = {K}")
ax.set_xlabel("Number of Clusters (K)", fontsize=11)
ax.set_ylabel("Total Distortion Squared (TD²)", fontsize=11)
ax.set_title("Elbow Curve  –  K-Means++ on A3 Dataset", fontsize=13, fontweight="bold")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(fontsize=10)
ax.set_xticks(list(ELBOW_RANGE))
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig10_elbow.png"))
plt.close(fig)

# ── Fig 11 : GMM BIC / AIC (for model selection context) ─────
print("[Fig 11] GMM BIC/AIC for different K …")
gmm_k_range = list(range(10, 71, 10))
bic_vals, aic_vals = [], []
for gk in gmm_k_range:
    gm = GaussianMixture(gk, covariance_type="full", n_init=3,
                         max_iter=200, random_state=SEED)
    gm.fit(X)
    bic_vals.append(gm.bic(X))
    aic_vals.append(gm.aic(X))
    print(f"  K={gk}  BIC={gm.bic(X):.4e}  AIC={gm.aic(X):.4e}")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(gmm_k_range, bic_vals, "s-", color="#E63946", linewidth=2,
        markersize=7, label="BIC (lower = better)")
ax.plot(gmm_k_range, aic_vals, "^-", color="#2A9D8F", linewidth=2,
        markersize=7, label="AIC (lower = better)")
ax.axvline(K, color="grey", linestyle="--", linewidth=1.5, label=f"True K = {K}")
ax.set_xlabel("Number of Components (K)", fontsize=11)
ax.set_ylabel("Information Criterion", fontsize=11)
ax.set_title("GMM Model Selection – BIC & AIC", fontsize=13, fontweight="bold")
ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
ax.legend(fontsize=10)
ax.set_xticks(gmm_k_range)
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig11_gmm_bic_aic.png"))
plt.close(fig)

# ── Fig 12 : Side-by-side 4-algorithm comparison ─────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for ax, (title, lbl, cents, td2_val, sil_val) in zip(axes.flat, ALGO_CONFIGS):
    ax.scatter(X[:, 0], X[:, 1], c=lbl, cmap="tab20", s=2,
               alpha=0.45, linewidths=0, vmin=0, vmax=K - 1)
    ax.scatter(cents[:, 0], cents[:, 1], c="white", edgecolors="black",
               s=80, zorder=5, linewidths=1.2, marker="*")
    ax.set_title(f"{title}\nTD²={td2_val:.3e}  Sil={sil_val:.4f}",
                 fontsize=9.5, fontweight="bold")
    ax.set_xlabel("x", fontsize=9); ax.set_ylabel("y", fontsize=9)
    ax.tick_params(labelsize=8)
fig.suptitle("Algorithm Comparison – A3 Benchmark (K=50)", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "fig12_comparison_grid.png"))
plt.close(fig)

print(f"\nAll figures saved to: {OUT_DIR}")
print("Analysis complete – run generate_pptx.py to build the presentation.")
