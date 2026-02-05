# -*- coding: utf-8 -*-
"""
predictor_display.py —— 只负责出图（论文风 / 高级可视化）

读取：
  predict/output/verification/preds_val_lgbm.csv
  predict/output/verification/preds_val_xgb.csv
  predict/output/verification/preds_val_ens.csv

输出（与你原来的文件名保持一致）：
  confusion_matrix_val_{lgbm,xgb,ens}.png
  roc_ovr_val_{lgbm,xgb,ens}.png
  pr_ovr_val_{lgbm,xgb,ens}.png
  p_true_hist_val_{lgbm,xgb,ens}.png
  p_true_violin_by_class_val_{lgbm,xgb,ens}.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    roc_curve, auc,
    precision_recall_curve, average_precision_score
)

# -------------------- 路径 --------------------
BASE_DIR  = os.path.dirname(__file__)          # .../predict
OUT_DIR   = os.path.join(BASE_DIR, "output")
VERIF_DIR = os.path.join(OUT_DIR, "verification")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(VERIF_DIR, exist_ok=True)

# -------------------- 风格与配色 --------------------
# Okabe-Ito（色盲友好，论文常用）
PALETTE = ["#0072B2", "#D55E00", "#009E73"]  # z1 z2 z3
GREY = "0.35"

def set_academic_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",

        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 11,

        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,

        "axes.grid": True,
        "grid.alpha": 0.18,
        "grid.linewidth": 0.8,

        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,

        "legend.frameon": False,
        "lines.linewidth": 2.2,
        "lines.solid_capstyle": "round",
    })

def save_fig(fig, path):
    fig.savefig(path, dpi=360, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[PLOT] {path}")

# -------------------- 读取 preds --------------------
def load_preds_val(suffix: str) -> pd.DataFrame:
    path = os.path.join(VERIF_DIR, f"preds_val_{suffix}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df

def get_y_pred_for_plot(df: pd.DataFrame) -> np.ndarray:
    # 优先用阈值优化后的预测
    if "y_pred_thresh" in df.columns:
        return df["y_pred_thresh"].values.astype(int)
    if "y_pred" in df.columns:
        return df["y_pred"].values.astype(int)
    if "y_pred_nominal" in df.columns:
        return df["y_pred_nominal"].values.astype(int)
    # 兜底：用概率 argmax
    P = df[["prob_z1", "prob_z2", "prob_z3"]].values.astype(float)
    return (P.argmax(axis=1) + 1).astype(int)

def get_proba(df: pd.DataFrame) -> np.ndarray:
    return df[["prob_z1", "prob_z2", "prob_z3"]].values.astype(float)

# -------------------- 图 1: Confusion Matrix --------------------
def plot_confusion_matrix(y_true, y_pred, model_name: str, out_png: str):
    labels = [1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(float)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm), where=row_sum != 0)

    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized fraction", rotation=90)

    ax.set_title(f"Confusion matrix (validation) — {model_name}", pad=10)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(["z=1", "z=2", "z=3"])
    ax.set_yticklabels(["z=1", "z=2", "z=3"])

    ax.grid(False)
    ax.set_xticks(np.arange(-.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 3, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(3):
        for j in range(3):
            count = int(cm[i, j])
            frac = cm_norm[i, j]
            ax.text(j, i, f"{count}\n({frac*100:.1f}%)", ha="center", va="center", fontsize=10)

    save_fig(fig, out_png)

# -------------------- 图 2/3: ROC / PR（更高级的 step + 填充） --------------------
def plot_roc_ovr(y_true, proba, model_name: str, out_png: str):
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    ax.plot([0, 1], [0, 1], linestyle="--", color=GREY, linewidth=1.2, label="Chance")

    for k, z in enumerate([1, 2, 3]):
        yb = (y_true == z).astype(int)
        pz = proba[:, z - 1]
        fpr, tpr, _ = roc_curve(yb, pz)
        roc_auc = auc(fpr, tpr)

        ax.step(fpr, tpr, where="post", color=PALETTE[k], label=f"z={z}  AUC={roc_auc:.3f}")
        ax.fill_between(fpr, 0, tpr, step="post", color=PALETTE[k], alpha=0.10)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"ROC (OvR) — validation — {model_name}", pad=10)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    save_fig(fig, out_png)

def _plot_iso_f1(ax, f1_values=(0.2, 0.4, 0.6, 0.8), color="0.70"):
    # PR 平面中的 iso-F1 曲线：F1 = 2PR/(P+R)
    # 约束：P = (F1 * R) / (2R - F1)
    r = np.linspace(1e-4, 1.0, 400)
    for f1 in f1_values:
        denom = (2 * r - f1)
        p = np.where(denom > 1e-6, (f1 * r) / denom, np.nan)
        p[(p < 0) | (p > 1)] = np.nan
        ax.plot(r, p, linestyle=":", linewidth=1.0, color=color, alpha=0.7)
        # 小标签
        idx = np.nanargmin(np.abs(r - 0.92))
        if np.isfinite(p[idx]):
            ax.text(r[idx], p[idx], f"F1={f1:.1f}", fontsize=9, color=color, ha="left", va="bottom")

def plot_pr_ovr(y_true, proba, model_name: str, out_png: str):
    fig, ax = plt.subplots(figsize=(7.4, 5.8))
    _plot_iso_f1(ax)

    for k, z in enumerate([1, 2, 3]):
        yb = (y_true == z).astype(int)
        pz = proba[:, z - 1]
        pre, rec, _ = precision_recall_curve(yb, pz)
        ap = average_precision_score(yb, pz)

        ax.step(rec, pre, where="post", color=PALETTE[k], label=f"z={z}  AP={ap:.3f}")
        ax.fill_between(rec, pre, step="post", color=PALETTE[k], alpha=0.10)

        base = float(yb.mean()) if yb.size else 0.0
        ax.hlines(base, 0, 1, colors=PALETTE[k], linestyles="--", linewidth=1.0, alpha=0.35)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"Precision–Recall (OvR) — validation — {model_name}", pad=10)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    save_fig(fig, out_png)

# -------------------- 图 4: 置信度 KDE（比柱状图高级） --------------------
def _silverman_bw(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return 0.1
    std = np.std(x, ddof=1)
    iqr = np.subtract(*np.percentile(x, [75, 25]))
    sigma = min(std, iqr / 1.349) if iqr > 1e-12 else std
    if sigma <= 1e-12:
        sigma = std if std > 1e-12 else 0.1
    bw = 0.9 * sigma * (n ** (-1/5))
    return float(max(bw, 1e-3))

def kde_1d(x, grid, bw=None):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.zeros_like(grid)
    if bw is None:
        bw = _silverman_bw(x)
    # 高斯核 KDE（纯 numpy）
    diff = (grid[:, None] - x[None, :]) / bw
    dens = np.exp(-0.5 * diff**2).sum(axis=1) / (x.size * bw * np.sqrt(2*np.pi))
    return dens

def plot_confidence_kde(y_true, proba, model_name: str, out_png: str):
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)
    p_true = proba[np.arange(len(y_true)), y_true - 1]
    p_true = np.clip(p_true, 0.0, 1.0)

    grid = np.linspace(0, 1, 400)
    dens = kde_1d(p_true, grid)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(grid, dens, color=GREY, linewidth=2.2)
    ax.fill_between(grid, 0, dens, color=GREY, alpha=0.12)

    mu = float(np.mean(p_true)) if p_true.size else 0.0
    med = float(np.median(p_true)) if p_true.size else 0.0
    ax.axvline(mu, color="0.15", linestyle="-", linewidth=1.6, label=f"Mean={mu:.3f}")
    ax.axvline(med, color="0.15", linestyle="--", linewidth=1.6, label=f"Median={med:.3f}")

    # rug（少量点，避免太花）
    if p_true.size > 0:
        idx = np.linspace(0, p_true.size - 1, min(60, p_true.size)).astype(int)
        ax.plot(p_true[idx], np.zeros_like(idx), "|", color="0.2", markersize=9, alpha=0.35)

    ax.set_title(f"Confidence distribution: p(true class) — {model_name}", pad=10)
    ax.set_xlabel("p(z_true)")
    ax.set_ylabel("Density")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper left")
    save_fig(fig, out_png)

# -------------------- 图 5: violin + box + jitter（更现代） --------------------
def plot_confidence_by_class_raincloud(y_true, proba, model_name: str, out_png: str):
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba, dtype=float)

    data = []
    for z in [1, 2, 3]:
        arr = proba[y_true == z, z - 1]
        arr = np.clip(arr, 0.0, 1.0)
        data.append(arr)

    fig, ax = plt.subplots(figsize=(7.4, 5.0))

    # violin（更柔和的边）
    parts = ax.violinplot(
        data, positions=[1, 2, 3],
        showmeans=False, showmedians=False, showextrema=False,
        widths=0.92
    )
    for i, b in enumerate(parts["bodies"]):
        b.set_facecolor(PALETTE[i])
        b.set_alpha(0.22)
        b.set_edgecolor("0.25")
        b.set_linewidth(1.0)

    # box（叠加，论文常见）
    bp = ax.boxplot(
        data, positions=[1, 2, 3],
        widths=0.26, patch_artist=True, showfliers=False,
        medianprops=dict(color="0.15", linewidth=2.0),
        boxprops=dict(linewidth=1.1, color="0.25"),
        whiskerprops=dict(linewidth=1.1, color="0.25"),
        capprops=dict(linewidth=1.1, color="0.25")
    )
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(PALETTE[i])
        box.set_alpha(0.35)

    # jitter（少量采样避免太乱）
    rng = np.random.default_rng(42)
    for i, arr in enumerate(data, start=1):
        if arr.size == 0:
            continue
        n = int(min(140, arr.size))
        samp = arr if arr.size <= n else rng.choice(arr, size=n, replace=False)
        xj = i + rng.normal(0, 0.045, size=samp.size)
        ax.scatter(xj, samp, s=10, color="0.2", alpha=0.22, linewidths=0)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["z=1", "z=2", "z=3"])
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("p(z_true)")
    ax.set_title(f"Confidence by true label — {model_name}", pad=10)
    save_fig(fig, out_png)

# -------------------- 总控 --------------------
def render_one_model(suffix: str, model_name: str):
    df = load_preds_val(suffix)
    if df.empty:
        print(f"[SKIP] preds_val_{suffix}.csv not found")
        return

    # 必要列检查
    need_cols = {"y_true", "prob_z1", "prob_z2", "prob_z3"}
    if not need_cols.issubset(set(df.columns)):
        print(f"[SKIP] preds_val_{suffix}.csv missing columns: {sorted(list(need_cols - set(df.columns)))}")
        return

    y_true = df["y_true"].values.astype(int)
    y_pred = get_y_pred_for_plot(df)
    proba = get_proba(df)

    # 你点名的 12 张图（每个模型 4 张）
    plot_confusion_matrix(
        y_true, y_pred, model_name=model_name,
        out_png=os.path.join(OUT_DIR, f"confusion_matrix_val_{suffix}.png")
    )
    plot_pr_ovr(
        y_true, proba, model_name=model_name,
        out_png=os.path.join(OUT_DIR, f"pr_ovr_val_{suffix}.png")
    )
    plot_roc_ovr(
        y_true, proba, model_name=model_name,
        out_png=os.path.join(OUT_DIR, f"roc_ovr_val_{suffix}.png")
    )
    plot_confidence_kde(
        y_true, proba, model_name=model_name,
        out_png=os.path.join(OUT_DIR, f"p_true_hist_val_{suffix}.png")
    )
    plot_confidence_by_class_raincloud(
        y_true, proba, model_name=model_name,
        out_png=os.path.join(OUT_DIR, f"p_true_violin_by_class_val_{suffix}.png")
    )

def main():
    set_academic_style()

    print("[Display Paths]")
    print("  OUT_DIR  :", OUT_DIR)
    print("  VERIF_DIR:", VERIF_DIR)

    render_one_model("lgbm", "LightGBM")
    render_one_model("xgb",  "XGBoost")
    render_one_model("ens",  "Ensemble")

    print("\n[Done] ✓ predictor_display.py finished plotting.")
    return

if __name__ == "__main__":
    main()
