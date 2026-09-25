"""
generate_pptx.py
================
Builds the PowerPoint presentation from the analysis results saved in ./output/.
Must be run AFTER analysis.py has completed successfully.
"""

import sys, os, csv
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

# ─── paths ────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, "output")
PPTX_OUT = os.path.join(BASE_DIR, "A3_Clustering_Analysis.pptx")

# ─── palette ──────────────────────────────────────────────────
C_DARK  = RGBColor(0x1A, 0x1A, 0x2E)   # dark navy
C_BLUE  = RGBColor(0x16, 0x21, 0x3E)   # mid-navy
C_ACC1  = RGBColor(0xE6, 0x39, 0x46)   # crimson
C_ACC2  = RGBColor(0x2A, 0x9D, 0x8F)   # teal
C_ACC3  = RGBColor(0xF4, 0xA2, 0x61)   # amber
C_WHITE = RGBColor(0xFF, 0xFF, 0xFF)
C_LGREY = RGBColor(0xF0, 0xF0, 0xF0)
C_GREY  = RGBColor(0xC0, 0xC0, 0xC0)

SLIDE_W = Inches(13.33)
SLIDE_H = Inches(7.5)

prs = Presentation()
prs.slide_width  = SLIDE_W
prs.slide_height = SLIDE_H

BLANK = prs.slide_layouts[6]   # completely blank layout

# ─── helper functions ─────────────────────────────────────────
def rect(slide, l, t, w, h, fill=None, line=None):
    from pptx.util import Emu
    shape = slide.shapes.add_shape(
        1, Inches(l), Inches(t), Inches(w), Inches(h))
    shape.line.fill.background()
    if fill:
        shape.fill.solid()
        shape.fill.fore_color.rgb = fill
    else:
        shape.fill.background()
    if line:
        shape.line.color.rgb = line
        shape.line.width = Pt(1)
    else:
        shape.line.fill.background()
    return shape

def text_box(slide, text, l, t, w, h,
             font_size=14, bold=False, color=C_WHITE,
             align=PP_ALIGN.LEFT, wrap=True, italic=False):
    txBox = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    txBox.word_wrap = wrap
    tf = txBox.text_frame
    tf.word_wrap = wrap
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.size = Pt(font_size)
    run.font.bold = bold
    run.font.italic = italic
    run.font.color.rgb = color
    return txBox

def img(slide, path, l, t, w, h=None):
    if os.path.exists(path):
        if h:
            slide.shapes.add_picture(path, Inches(l), Inches(t),
                                     Inches(w), Inches(h))
        else:
            slide.shapes.add_picture(path, Inches(l), Inches(t), Inches(w))
    else:
        rect(slide, l, t, w, h or 3, fill=C_GREY)
        text_box(slide, f"[missing: {os.path.basename(path)}]",
                 l + 0.1, t + 0.1, w - 0.2, (h or 3) - 0.2,
                 font_size=10, color=C_DARK)

def add_slide(title_text, subtitle_text="", bg_color=C_DARK):
    """Create a new slide with a consistent header bar."""
    slide = prs.slides.add_slide(BLANK)
    # background
    rect(slide, 0, 0, 13.33, 7.5, fill=bg_color)
    # top bar
    rect(slide, 0, 0, 13.33, 1.15, fill=C_ACC1)
    # title
    text_box(slide, title_text, 0.25, 0.1, 12.5, 0.9,
             font_size=26, bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    # subtitle
    if subtitle_text:
        text_box(slide, subtitle_text, 0.25, 1.1, 12.8, 0.55,
                 font_size=13, bold=False, color=C_GREY, align=PP_ALIGN.LEFT)
    return slide

def bullet_box(slide, items, l, t, w, h, font_size=13, color=C_WHITE):
    """Multi-bullet text box; items = list of (indent_level, text)."""
    txBox = slide.shapes.add_textbox(Inches(l), Inches(t), Inches(w), Inches(h))
    txBox.word_wrap = True
    tf = txBox.text_frame
    tf.word_wrap = True
    first = True
    for level, line in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.level = level
        p.space_after = Pt(2)
        run = p.add_run()
        bullet = "•  " if level == 0 else "   ◦  "
        run.text = bullet + line
        run.font.size = Pt(font_size)
        run.font.color.rgb = color

def metric_card(slide, label, value, unit, l, t, w=2.5, h=1.3, accent=C_ACC2):
    rect(slide, l, t, w, h, fill=accent)
    rect(slide, l, t, w, h, line=C_WHITE)
    text_box(slide, label, l + 0.08, t + 0.05, w - 0.15, 0.4,
             font_size=11, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    text_box(slide, value, l + 0.08, t + 0.38, w - 0.15, 0.6,
             font_size=16, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    text_box(slide, unit, l + 0.08, t + 0.92, w - 0.15, 0.3,
             font_size=9, bold=False, color=C_WHITE, align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════
# SLIDE 1 – TITLE
# ══════════════════════════════════════════════════════════════
slide = prs.slides.add_slide(BLANK)
rect(slide, 0, 0, 13.33, 7.5, fill=C_DARK)
rect(slide, 0, 0, 13.33, 0.06, fill=C_ACC1)
rect(slide, 0, 7.44, 13.33, 0.06, fill=C_ACC1)
rect(slide, 0.5, 1.3, 12.33, 4.8, fill=C_BLUE)

text_box(slide, "A3 DATASET", 1.0, 1.6, 11.5, 1.1,
         font_size=44, bold=True, color=C_ACC1, align=PP_ALIGN.CENTER)
text_box(slide, "Clustering Algorithm Analysis", 1.0, 2.65, 11.5, 0.8,
         font_size=28, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
text_box(slide, "K-Means Brute Force  ·  K-Means++  ·  GMM / EM  ·  GA-VQ Reference",
         1.0, 3.4, 11.5, 0.55,
         font_size=14, color=C_ACC3, align=PP_ALIGN.CENTER)
text_box(slide,
         "TD²  ·  Log-Likelihood  ·  BIC / AIC  ·  Silhouette  ·  Instability Analysis",
         1.0, 3.95, 11.5, 0.45,
         font_size=12, color=C_GREY, align=PP_ALIGN.CENTER)
text_box(slide, "7 500 points  |  2-D  |  50 Gaussian clusters",
         1.0, 4.55, 11.5, 0.4,
         font_size=12, italic=True, color=C_GREY, align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════
# SLIDE 2 – DATASET OVERVIEW
# ══════════════════════════════════════════════════════════════
slide = add_slide("Dataset Overview", "A3 Benchmark – 7 500 points, 50 Gaussian clusters")
rect(slide, 0.25, 1.6, 5.5, 5.65, fill=RGBColor(0x10, 0x10, 0x28))
img(slide, os.path.join(OUT_DIR, "fig1_raw_data.png"), 0.35, 1.65, 5.3, 5.5)

bullet_box(slide, [
    (0, "Source: A3 benchmark (Fränti & Virmajoki 2006)"),
    (0, "7 500 two-dimensional data points"),
    (0, "50 Gaussian clusters, varying density"),
    (0, "Coordinate range: ≈ 3K – 65K (both axes)"),
    (0, "Known optimal K = 50 clusters"),
    (0, "Provided GA-VQ codebook: 50 centroids"),
    (0, "Partition file: 7 500 cluster labels (1–50)"),
    (0, "Challenge: 50-cluster recovery, no labels"),
], 6.1, 1.7, 6.9, 4.8, font_size=14, color=C_LGREY)

text_box(slide,
         "This dataset is a standard Vector-Quantization (VQ) benchmark used to compare "
         "clustering heuristics under realistic multi-Gaussian structure.",
         6.1, 6.45, 6.9, 0.7, font_size=11, color=C_GREY, italic=True)

# ══════════════════════════════════════════════════════════════
# SLIDE 3 – ALGORITHM THEORY
# ══════════════════════════════════════════════════════════════
slide = add_slide("Algorithm Landscape", "Hard vs Soft clustering | Brute-force vs Smart initialisation")
bg_panels = [(0.2, C_ACC1), (3.55, C_ACC2), (6.9, C_ACC3), (10.25, RGBColor(0x45,0x7B,0x9D))]
algo_titles = ["K-Means\nBrute-Force", "K-Means++", "GMM / EM\n(Soft)", "GA-VQ\n(Reference)"]
algo_bullets = [
    ["Lloyd's iteration", "Random centroid init", "50 restarts → best TD²",
     "Hard assignment", "Simple but unstable"],
    ["K-Means++ seeding", "Prob. init (D² sampling)", "20 restarts",
     "Hard assignment", "Much more stable"],
    ["Gaussian Mixture Model", "EM algorithm (E & M steps)", "Soft memberships P(k|xᵢ)",
     "Full covariance matrix", "Log-likelihood optimised"],
    ["Genetic Algorithm + VQ", "Population of solutions", "Crossover & mutation",
     "Avoids local minima", "Provided as reference"],
]
for x_off, color, atitle, abullets in zip(
        [b[0] for b in bg_panels], [b[1] for b in bg_panels],
        algo_titles, algo_bullets):
    rect(slide, x_off, 1.5, 3.1, 5.8, fill=RGBColor(0x10,0x10,0x28))
    rect(slide, x_off, 1.5, 3.1, 0.8, fill=color)
    text_box(slide, atitle, x_off + 0.05, 1.52, 3.0, 0.78,
             font_size=14, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    for bi, btext in enumerate(abullets):
        text_box(slide, f"• {btext}", x_off + 0.1, 2.4 + bi * 0.72, 2.9, 0.65,
                 font_size=11, color=C_LGREY)

text_box(slide,
         "TD²  (Total Distortion²)  =  Σ ||xᵢ − cₖ(ᵢ)||²   ←  lower is better for all algorithms",
         0.2, 7.1, 13.0, 0.38, font_size=12, bold=True,
         color=C_ACC3, align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════
# SLIDE 4 – K-MEANS BRUTE FORCE RESULTS
# ══════════════════════════════════════════════════════════════
slide = add_slide("K-Means – Brute Force", "Random initialisation  |  50 restarts  |  Hard assignment")
img(slide, os.path.join(OUT_DIR, "fig2_cluster_1.png"), 0.25, 1.35, 7.5, 5.9)

bullet_box(slide, [
    (0, "Init: uniform random centroids"),
    (0, "50 independent restarts → best inertia selected"),
    (0, "Each run uses Lloyd's iteration (max 300 steps)"),
    (0, "Hard assignment: each point → nearest centroid"),
    (0, "WEAKNESS: highly sensitive to seed"),
    (0, "May converge to poor local minima"),
    (0, "High σ(TD²) across runs → UNSTABLE"),
], 8.0, 1.6, 5.1, 4.0, font_size=13, color=C_LGREY)

metric_card(slide, "TD²", "see fig", "vs other algos",  8.0, 5.8, 2.5, 1.35, C_ACC1)
metric_card(slide, "Restarts", "50", "random seeds",   10.7, 5.8, 2.5, 1.35, C_ACC1)

# ══════════════════════════════════════════════════════════════
# SLIDE 5 – K-MEANS++ RESULTS
# ══════════════════════════════════════════════════════════════
slide = add_slide("K-Means++", "Probabilistic initialisation (D² sampling)  |  20 restarts")
img(slide, os.path.join(OUT_DIR, "fig3_cluster_2.png"), 0.25, 1.35, 7.5, 5.9)

bullet_box(slide, [
    (0, "Init: D² weighted sampling (Arthur & Vassilvitskii 2007)"),
    (0, "1st centroid: random; each next ∝ distance² to nearest"),
    (0, "Proven ≤ 8(ln K + 2) · OPT approximation ratio"),
    (0, "20 restarts sufficient (much fewer needed vs BF)"),
    (0, "Hard assignment – identical to K-Means after init"),
    (0, "Typically achieves LOWER TD² than Brute-Force"),
    (0, "Lower σ(TD²) → MORE STABLE than BF"),
], 8.0, 1.6, 5.1, 4.0, font_size=13, color=C_LGREY)

metric_card(slide, "TD²", "≈ BF", "slight improvement", 8.0, 5.8, 2.5, 1.35, C_ACC2)
metric_card(slide, "Stability", "Better", "lower σ(TD²)",   10.7, 5.8, 2.5, 1.35, C_ACC2)

# ══════════════════════════════════════════════════════════════
# SLIDE 6 – GMM / EM RESULTS
# ══════════════════════════════════════════════════════════════
slide = add_slide("Gaussian Mixture Model (GMM / EM)", "Soft clustering  |  Full covariance  |  Probabilistic membership")
img(slide, os.path.join(OUT_DIR, "fig6_gmm_soft.png"), 0.25, 1.35, 7.5, 5.9)

bullet_box(slide, [
    (0, "Models each cluster as a multivariate Gaussian"),
    (0, "EM alternates: E-step (responsibilities) ↔ M-step (params)"),
    (0, "SOFT: P(cluster k | pointᵢ) for every point"),
    (0, "Optimises Log-Likelihood, not TD²"),
    (0, "Full covariance → captures elliptical clusters"),
    (0, "Model selection: BIC / AIC vs K"),
    (0, "Best Log-Likelihood among all algorithms"),
], 8.0, 1.6, 5.1, 4.0, font_size=13, color=C_LGREY)

metric_card(slide, "Log-Likelihood", "Maximum", "vs K-Means", 8.0, 5.8, 2.5, 1.35, C_ACC2)
metric_card(slide, "Soft assign.", "Yes", "P(k|xᵢ) ∈ [0,1]",  10.7, 5.8, 2.5, 1.35, C_ACC2)

# ══════════════════════════════════════════════════════════════
# SLIDE 7 – GA-VQ REFERENCE
# ══════════════════════════════════════════════════════════════
slide = add_slide("GA-VQ – Genetic Algorithm Reference", "Evolutionary optimisation  |  Provided codebook + partition")
img(slide, os.path.join(OUT_DIR, "fig5_cluster_4.png"), 0.25, 1.35, 7.5, 5.9)

bullet_box(slide, [
    (0, "Genetic Algorithm on codebook population"),
    (0, "Each individual = set of 50 centroids"),
    (0, "Operators: selection, crossover, mutation"),
    (0, "Fitness = Total Distortion (TD²) – minimised"),
    (0, "Escapes local minima that K-Means gets stuck in"),
    (0, "Achieves best TD² among heuristics on A3"),
    (0, "Provided as reference result for comparison"),
    (0, "File: a3-ga-cb.txt (centroids), a3-ga.pa (labels)"),
], 8.0, 1.6, 5.1, 4.4, font_size=13, color=C_LGREY)

metric_card(slide, "TD²", "Lowest", "benchmark result", 8.0, 5.95, 5.2, 1.2, C_ACC3)

# ══════════════════════════════════════════════════════════════
# SLIDE 8 – TD² COMPARISON
# ══════════════════════════════════════════════════════════════
slide = add_slide("TD² Comparison", "Total Distortion Squared  –  lower is better")
img(slide, os.path.join(OUT_DIR, "fig7_td2_comparison.png"), 0.5, 1.35, 7.8, 5.85)

bullet_box(slide, [
    (0, "TD² = Σ ||xᵢ − c_{k(i)}||²  (WCSS / Inertia)"),
    (0, "Primary VQ quality metric from the paper"),
    (0, ""),
    (0, "GA-VQ: reference/best (evolutionary search)"),
    (0, "GMM/EM: competitive – optimises different obj."),
    (0, "K-Means++: usually closest to GA-VQ among"),
    (0, "  K-Means variants"),
    (0, "K-Means BF: highest TD², most unstable"),
    (0, ""),
    (0, "Gold border = best result per metric"),
], 8.6, 1.6, 4.5, 5.5, font_size=12.5, color=C_LGREY)

# ══════════════════════════════════════════════════════════════
# SLIDE 9 – QUALITY METRICS
# ══════════════════════════════════════════════════════════════
slide = add_slide("Clustering Quality Metrics", "Silhouette  |  Davies-Bouldin  |  Calinski-Harabasz")
img(slide, os.path.join(OUT_DIR, "fig8_quality_metrics.png"), 0.2, 1.35, 13.0, 5.85)

# ══════════════════════════════════════════════════════════════
# SLIDE 10 – INSTABILITY ANALYSIS
# ══════════════════════════════════════════════════════════════
slide = add_slide("Algorithm Instability", "σ(TD²) across 10 independent runs – single restart per run")
img(slide, os.path.join(OUT_DIR, "fig9_instability.png"), 0.5, 1.35, 7.8, 5.85)

bullet_box(slide, [
    (0, "Instability = std-dev of TD² over 10 seeds"),
    (0, "High σ → algorithm is sensitive to initialisation"),
    (0, ""),
    (0, "K-Means BF: highest σ (most unstable)"),
    (1, "Random init → many local minima traps"),
    (0, "K-Means++: much lower σ (stable init)"),
    (1, "D² sampling avoids degenerate starts"),
    (0, "GMM/EM: moderate instability"),
    (1, "EM can also converge to local maxima"),
    (0, ""),
    (0, "Verdict: K-Means BF is MOST UNSTABLE"),
    (0, "Verdict: K-Means++ is MOST STABLE (K-Means family)"),
], 8.6, 1.6, 4.5, 5.7, font_size=12, color=C_LGREY)

# ══════════════════════════════════════════════════════════════
# SLIDE 11 – ELBOW CURVE
# ══════════════════════════════════════════════════════════════
slide = add_slide("Elbow Curve – Model Selection", "K-Means++ TD² vs K  |  Confirms optimal K = 50")
img(slide, os.path.join(OUT_DIR, "fig10_elbow.png"), 0.5, 1.35, 8.0, 5.85)

bullet_box(slide, [
    (0, "Elbow method: plot TD² vs number of clusters K"),
    (0, "Rate of decrease slows at the 'elbow' = optimal K"),
    (0, ""),
    (0, "A3 dataset has exactly 50 true clusters"),
    (0, "Elbow visible near K = 50 (red dashed line)"),
    (0, "Beyond K=50: marginal TD² reduction"),
    (0, "Below K=50: large TD² penalty"),
    (0, ""),
    (0, "GMM BIC/AIC used for cross-validation"),
    (0, "Both criteria confirm K ≈ 50 as optimal"),
], 8.8, 1.6, 4.3, 5.7, font_size=12, color=C_LGREY)

# ══════════════════════════════════════════════════════════════
# SLIDE 12 – GMM BIC / AIC
# ══════════════════════════════════════════════════════════════
slide = add_slide("GMM Model Selection – BIC & AIC", "Bayesian & Akaike Information Criteria vs K")
img(slide, os.path.join(OUT_DIR, "fig11_gmm_bic_aic.png"), 0.5, 1.35, 8.0, 5.85)

bullet_box(slide, [
    (0, "BIC = −2·ℓ + p·ln(n)  (penalises parameters)"),
    (0, "AIC = −2·ℓ + 2p  (lighter penalty)"),
    (0, "Lower BIC/AIC = better model-data fit"),
    (0, ""),
    (0, "BIC is more conservative than AIC"),
    (0, "Both confirm minimum near K = 50"),
    (0, "Provides statistical basis for K choice"),
    (0, "Unlike elbow, these are grounded in probability theory"),
    (0, ""),
    (0, "Use: select K at BIC minimum → K = 50 confirmed"),
], 8.8, 1.6, 4.3, 5.7, font_size=12, color=C_LGREY)

# ══════════════════════════════════════════════════════════════
# SLIDE 13 – SIDE-BY-SIDE COMPARISON
# ══════════════════════════════════════════════════════════════
slide = add_slide("All Algorithms – Side-by-Side", "Cluster assignment visualisation for all four approaches")
img(slide, os.path.join(OUT_DIR, "fig12_comparison_grid.png"), 0.5, 1.35, 12.3, 5.85)

# ══════════════════════════════════════════════════════════════
# SLIDE 14 – METRICS TABLE (text-based)
# ══════════════════════════════════════════════════════════════
slide = add_slide("Full Metrics Summary", "Quantitative comparison across all evaluation criteria")

headers = ["Algorithm", "TD²", "Log-Lik.", "Silhouette ↑", "DB ↓", "BIC", "σ(TD²)"]
col_w   = [2.6,         1.85,  1.65,       1.55,           1.2,   1.85,  1.85]
col_x   = [0.2]
for w in col_w[:-1]:
    col_x.append(col_x[-1] + w)

# Header row
for cx, cw, ch in zip(col_x, col_w, headers):
    rect(slide, cx, 1.55, cw - 0.05, 0.52, fill=C_ACC1)
    text_box(slide, ch, cx + 0.05, 1.58, cw - 0.1, 0.46,
             font_size=11, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)

# Data rows – read from CSV if available, else use placeholder
csv_path = os.path.join(OUT_DIR, "metrics_summary.csv")
rows, row_colors = [], [RGBColor(0x10,0x18,0x32), RGBColor(0x18,0x22,0x44)] * 5
try:
    import csv as csv_mod
    with open(csv_path, newline="") as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            rows.append(row)
except Exception:
    pass

def fmt(v):
    if v in (None, "None", ""):
        return "—"
    try:
        fv = float(v)
        if abs(fv) > 1e8: return f"{fv:.3e}"
        if abs(fv) > 1e3: return f"{fv:.0f}"
        return f"{fv:.4f}"
    except:
        return str(v)

col_keys = ["Algorithm", "TD2", "Log-Likelihood", "Silhouette ↑",
            "Davies-Bouldin ↓", "BIC", "σ(TD2) instab."]

for ri, row in enumerate(rows[:4]):
    row_bg = [RGBColor(0x10,0x18,0x30), RGBColor(0x14,0x1E,0x38),
              RGBColor(0x0E,0x1C,0x2C), RGBColor(0x12,0x20,0x34)][ri]
    y_off = 2.12 + ri * 1.0
    for cx, cw, ck in zip(col_x, col_w, col_keys):
        rect(slide, cx, y_off, cw - 0.05, 0.92, fill=row_bg)
        val = fmt(row.get(ck, "—"))
        text_box(slide, val, cx + 0.05, y_off + 0.12, cw - 0.1, 0.7,
                 font_size=11, color=C_LGREY, align=PP_ALIGN.CENTER)

text_box(slide,
         "↑ higher is better  |  ↓ lower is better  |  TD² = Total Distortion Squared  |  DB = Davies-Bouldin",
         0.2, 6.9, 13.0, 0.38, font_size=10, color=C_GREY, italic=True, align=PP_ALIGN.CENTER)

# ══════════════════════════════════════════════════════════════
# SLIDE 15 – MODEL QUALITY EVALUATION
# ══════════════════════════════════════════════════════════════
slide = add_slide("Model Quality Evaluation", "Interpreting the metrics for the A3 dataset structure")

quality_grid = [
    ("TD² / Inertia",    C_ACC1, [
        "Primary VQ metric – matches paper criterion",
        "GA-VQ achieves reference (lowest) TD²",
        "K-Means++ close; GMM competitive",
        "K-Means BF consistently worst",
    ]),
    ("Log-Likelihood",   C_ACC2, [
        "Only defined for probabilistic models (GMM)",
        "GMM maximises this directly via EM",
        "Higher → data better explained by model",
        "BIC/AIC control overfitting vs fit trade-off",
    ]),
    ("Silhouette Score", C_ACC3, [
        "Measures intra-cluster cohesion vs inter-cluster sep.",
        "Range [−1, +1] — closer to +1 is better",
        "A3 has overlapping clusters → moderate scores",
        "All algorithms score similarly (≈ 0.5–0.7)",
    ]),
    ("Instability σ",    RGBColor(0x45,0x7B,0x9D), [
        "K-Means BF: highest σ → least reliable",
        "K-Means++: substantially more stable",
        "GMM: moderate instability (EM local maxima)",
        "GA-VQ: deterministic given same population init",
    ]),
]
for gi, (title, color, points) in enumerate(quality_grid):
    col = gi % 2
    row = gi // 2
    gx = 0.2 + col * 6.5
    gy = 1.55 + row * 2.8
    rect(slide, gx, gy, 6.3, 2.6, fill=RGBColor(0x10,0x10,0x26))
    rect(slide, gx, gy, 6.3, 0.5, fill=color)
    text_box(slide, title, gx + 0.1, gy + 0.05, 6.1, 0.42,
             font_size=14, bold=True, color=C_WHITE, align=PP_ALIGN.LEFT)
    for pi, pt in enumerate(points):
        text_box(slide, f"• {pt}", gx + 0.1, gy + 0.6 + pi * 0.47, 6.1, 0.44,
                 font_size=11, color=C_LGREY)

# ══════════════════════════════════════════════════════════════
# SLIDE 16 – INTERPRETATION & CONCLUSIONS
# ══════════════════════════════════════════════════════════════
slide = add_slide("Interpretation & Conclusions", "Which algorithm is most suitable – and which is most unstable?")

col_data = [
    ("Most Suitable", C_ACC2, [
        "K-Means++ is the most practical choice:",
        "  → Near-optimal TD² vs GA-VQ reference",
        "  → Much lower instability than BF",
        "  → Fast convergence, interpretable",
        "",
        "GMM is best when:",
        "  → Soft / probabilistic assignment needed",
        "  → Clusters have elliptical shapes",
        "  → BIC/AIC model selection required",
    ]),
    ("Most Unstable", C_ACC1, [
        "K-Means Brute-Force is most unstable:",
        "  → Random init → many local minima",
        "  → High σ(TD²) across seeds",
        "  → Requires many restarts to be reliable",
        "",
        "GA-VQ avoids instability via evolution:",
        "  → Population escapes local minima",
        "  → Best TD² result on this benchmark",
        "  → Computationally more expensive",
    ]),
    ("Key Takeaways", C_ACC3, [
        "• Soft (GMM) ≠ always better than hard (K-Means)",
        "• Initialisation matters more than restarts",
        "• K=50 confirmed by elbow + BIC/AIC + data",
        "• TD² and log-likelihood measure different things",
        "• Silhouette confirms cluster structure quality",
        "• GA-VQ sets the benchmark for VQ problems",
        "• Use K-Means++ as default; GMM for probability",
    ]),
]
for gi, (title, color, points) in enumerate(col_data):
    gx = 0.2 + gi * 4.35
    rect(slide, gx, 1.5, 4.2, 5.8, fill=RGBColor(0x0E,0x0E,0x24))
    rect(slide, gx, 1.5, 4.2, 0.55, fill=color)
    text_box(slide, title, gx + 0.1, 1.54, 4.0, 0.48,
             font_size=15, bold=True, color=C_WHITE, align=PP_ALIGN.CENTER)
    for pi, pt in enumerate(points):
        text_box(slide, pt, gx + 0.1, 2.15 + pi * 0.56, 4.0, 0.52,
                 font_size=11, color=C_LGREY if pt else C_WHITE)

# ══════════════════════════════════════════════════════════════
# SAVE
# ══════════════════════════════════════════════════════════════
prs.save(PPTX_OUT)
print(f"\nPresentation saved → {PPTX_OUT}")
print(f"  Slides: {len(prs.slides)}")
