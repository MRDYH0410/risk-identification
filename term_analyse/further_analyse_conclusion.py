# -*- coding: utf-8 -*-
"""
further_analyse_conclusion.py

功能 1
  - 分组表 + 上下贴合双条形网格图

功能 2
  - 两个词云子图（SRC Positive / SRC Negative）
  - 圆形、高密、中心少数大词
  - 多元词轻微加权
  - Type1/2/3 颜色区分 + 图例
"""

import os
import re
import logging
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    from wordcloud import WordCloud
except Exception:
    WordCloud = None

from matplotlib import font_manager


# -----------------------------
# 路径配置
# -----------------------------
PROJ_ROOT = r"D:\UOW\risk idenfitication\code\risk-identification"

RES_TABLE_CSV = os.path.join(
    PROJ_ROOT, "term_analyse", "output", "importance", "table", "res_table_clean.csv"
)

CONC_OUT_DIR = os.path.join(PROJ_ROOT, "term_analyse", "output", "conclusion")
os.makedirs(CONC_OUT_DIR, exist_ok=True)

GROUP_TABLE_CSV = os.path.join(CONC_OUT_DIR, "rq2_term_group_table_by_z_gramtype.csv")
GROUP_FIG_PNG = os.path.join(CONC_OUT_DIR, "rq2_term_group_grid.png")
GROUP_FIG_PDF = os.path.join(CONC_OUT_DIR, "rq2_term_group_grid.pdf")

WC_FIG_PNG = os.path.join(CONC_OUT_DIR, "rq2_wordcloud_impdelta_posneg.png")
WC_FIG_PDF = os.path.join(CONC_OUT_DIR, "rq2_wordcloud_impdelta_posneg.pdf")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("further_analyse_conclusion")


# -----------------------------
# 功能 1 图参数
# -----------------------------
COLOR_DELTA = "#0072B2"
COLOR_SCORE = "#009E73"
PANEL_FACE = "#FCFCFC"
PANEL_EDGE = "#AAAAAA"

# 让 term 文本区域与柱子区域彻底分开，避免遮挡
TERM_X = 0.03
BAR_LEFT = 0.52          # 原来 0.38，改大以免挡住 term
MAX_BAR_WIDTH = 0.40     # 对应缩小，给右侧数值留空间
VALUE_PAD = 0.012        # 数值离柱子尾部的距离

# 轴范围稍微放宽，保证数值能贴在柱子外侧
XMAX = 1.06

BAR_H = 0.30
BAR_GAP = 0.02
OFFSET = (BAR_H + BAR_GAP) / 2.0


# -----------------------------
# 词云参数
# -----------------------------
WC_ABBR = "SRC"
WC_TITLE_POS = f"{WC_ABBR} Positive"
WC_TITLE_NEG = f"{WC_ABBR} Negative"

WC_SIZE = 900
WC_MAX_WORDS = 420
WC_MIN_FONT = 2
WC_MAX_FONT = 190
WC_MARGIN = 0
WC_RANDOM_STATE = 42

WC_WEIGHT_POWER = 1.8
WC_TOPN_PER_SIGN = 420

# 多元词轻微加权（一丢丢）
WC_MULTIWORD_BOOST = 1.08

# Type 颜色
TYPE_COLOR = {
    "1": "#0072B2",
    "2": "#D55E00",
    "3": "#009E73",
}
TYPE_FALLBACK = "#111111"


# -----------------------------
# 工具函数
# -----------------------------
def _safe_zscore(series: pd.Series) -> pd.Series:
    col = series.astype(float)
    mean = col.mean()
    std = col.std()
    if std <= 1e-6:
        std = 1.0
    return (col - mean) / std


def _get_default_font_path() -> str:
    try:
        fp = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans"))
        if fp and os.path.isfile(fp):
            return fp
    except Exception:
        pass
    return ""


def _make_circle_mask(size: int = 900) -> np.ndarray:
    h = w = int(size)
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    r = int(min(h, w) * 0.48)

    dist2 = (x - cx) ** 2 + (y - cy) ** 2
    mask = np.ones((h, w), dtype=np.uint8) * 255
    mask[dist2 <= r * r] = 0
    return mask


def _norm_risk_type(val) -> str:
    s = str(val).strip()
    m = re.search(r"([123])", s)
    return m.group(1) if m else "NA"


def _format_type_title(val) -> str:
    d = _norm_risk_type(val)
    return f"Type {d}" if d in {"1", "2", "3"} else "Type"


def _build_wordcloud_freq_and_type(
    df: pd.DataFrame, sign: str, topn: int
) -> Tuple[Dict[str, float], Dict[str, str]]:
    sub = df.copy()
    sub["term"] = sub["term"].astype(str)

    sub["imp_delta"] = pd.to_numeric(sub["imp_delta"], errors="coerce").fillna(0.0)
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=["term", "imp_delta"])

    if "risk_type_z" not in sub.columns:
        sub["risk_type_z"] = "NA"
    sub["risk_type_z"] = sub["risk_type_z"].apply(_norm_risk_type)

    if sign == "pos":
        sub = sub[sub["imp_delta"] > 0].copy()
        sub["base_w"] = sub["imp_delta"].astype(float)
    else:
        sub = sub[sub["imp_delta"] < 0].copy()
        sub["base_w"] = (-sub["imp_delta"]).astype(float)

    if sub.empty:
        return {}, {}

    sub["n_tokens"] = sub["term"].str.split().str.len().fillna(0).astype(int)
    sub["mw_boost"] = np.where(sub["n_tokens"] > 1, WC_MULTIWORD_BOOST, 1.0)
    sub["base_w"] = sub["base_w"] * sub["mw_boost"]

    agg = (
        sub.groupby(["term", "risk_type_z"], as_index=False)["base_w"]
        .sum()
        .rename(columns={"base_w": "w"})
    )

    agg = agg.sort_values(["term", "w"], ascending=[True, False])
    dominant = agg.drop_duplicates("term", keep="first")[["term", "risk_type_z"]].copy()

    total_w = agg.groupby("term", as_index=False)["w"].sum().rename(columns={"w": "w_total"})
    term_df = total_w.merge(dominant, on="term", how="left")
    term_df = term_df.sort_values("w_total", ascending=False).head(topn).copy()

    w = term_df["w_total"].values.astype(float)
    w = np.maximum(w, 1e-12)
    w = (w / (np.max(w) + 1e-12)) ** WC_WEIGHT_POWER
    w = np.maximum(w, 0.01)

    freq: Dict[str, float] = {}
    type_map: Dict[str, str] = {}
    for term, ww, z in zip(term_df["term"].tolist(), w.tolist(), term_df["risk_type_z"].tolist()):
        t = str(term).strip()
        if not t:
            continue
        freq[t] = float(ww)
        type_map[t] = _norm_risk_type(z)

    return freq, type_map


# -----------------------------
# 功能 1：分组表 + 网格图
# -----------------------------
def build_group_table(top_k: int = 10) -> pd.DataFrame:
    if not os.path.isfile(RES_TABLE_CSV):
        logger.error(f"res_table_clean.csv not found: {RES_TABLE_CSV}")
        return pd.DataFrame()

    df = pd.read_csv(RES_TABLE_CSV)
    required_cols: List[str] = [
        "company", "risk_type_z", "term",
        "imp_early", "imp_late", "imp_delta", "term_score_mean",
        "mode_speaker_early", "mode_speaker_late",
        "mode_dominant_cause_early", "mode_dominant_cause_late",
    ]
    for c in required_cols:
        if c not in df.columns:
            logger.error(f"Required column '{c}' not found in res_table_clean.")
            return pd.DataFrame()

    df["term"] = df["term"].astype(str)
    df["n_tokens"] = df["term"].str.split().str.len().fillna(0).astype(int)
    df["gram_type"] = df["n_tokens"].apply(lambda n: "unigram" if n == 1 else "multiword")

    for c in ["imp_early", "imp_late", "imp_delta", "term_score_mean"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    df["abs_imp_delta"] = df["imp_delta"].abs()
    df["abs_imp_delta_z"] = _safe_zscore(df["abs_imp_delta"])
    df["term_score_mean_z"] = _safe_zscore(df["term_score_mean"])
    df["sensitivity"] = df["abs_imp_delta_z"] - df["term_score_mean_z"]

    pieces = []
    for (z_type, gram), sub in df.groupby(["risk_type_z", "gram_type"]):
        sub = sub.sort_values(by=["sensitivity", "abs_imp_delta"], ascending=[False, False]).copy()
        sub["rank_in_group"] = np.arange(1, len(sub) + 1, dtype=int)
        pieces.append(sub.head(top_k))

    if not pieces:
        return pd.DataFrame()

    res = pd.concat(pieces, ignore_index=True)
    out_cols = [
        "risk_type_z", "gram_type", "rank_in_group",
        "company", "term",
        "imp_early", "imp_late", "imp_delta", "abs_imp_delta",
        "term_score_mean", "sensitivity",
        "mode_speaker_early", "mode_speaker_late",
        "mode_dominant_cause_early", "mode_dominant_cause_late",
    ]
    res = res[out_cols].sort_values(by=["risk_type_z", "gram_type", "rank_in_group"])
    res.to_csv(GROUP_TABLE_CSV, index=False)
    logger.info(f"[save] {GROUP_TABLE_CSV}")
    return res


def _panel_norm_params(sub: pd.DataFrame) -> Tuple[float, float, float]:
    """
    每个子图内部单独归一化
    返回 abs_delta_max, score_shift, score_max_shifted
    """
    abs_max = float(sub["abs_imp_delta"].astype(float).max()) if len(sub) else 0.0
    score = sub["term_score_mean"].astype(float)
    score_min = float(score.min()) if len(score) else 0.0
    score_max = float(score.max()) if len(score) else 0.0

    score_shift = -score_min if score_min < 0 else 0.0
    score_max_shifted = score_max + score_shift
    return abs_max, score_shift, score_max_shifted


def plot_group_grid(group_df: pd.DataFrame, top_k: int = 10) -> None:
    if group_df.empty:
        return

    df = group_df.copy()
    df = df[df["rank_in_group"] <= top_k].copy()

    # 统一 risk_type 展示与排序
    df["risk_type_norm"] = df["risk_type_z"].apply(_norm_risk_type)
    z_list = [z for z in ["1", "2", "3"] if z in df["risk_type_norm"].unique().tolist()]
    if not z_list:
        z_list = sorted(df["risk_type_norm"].unique().tolist())

    gram_types_all = ["unigram", "multiword"]
    n_rows, n_cols = len(gram_types_all), len(z_list)
    if n_rows == 0 or n_cols == 0:
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 3.2 * n_rows), squeeze=False)

    for row_idx, gram in enumerate(gram_types_all):
        for col_idx, z in enumerate(z_list):
            ax = axes[row_idx, col_idx]
            sub = df[(df["gram_type"] == gram) & (df["risk_type_norm"] == z)].copy()
            sub = sub.sort_values("rank_in_group")
            n_terms = len(sub)

            ax.set_facecolor(PANEL_FACE)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)
                spine.set_edgecolor(PANEL_EDGE)

            ax.set_xlim(0.0, XMAX)
            ax.set_xticks([])
            ax.set_yticks([])

            if n_terms == 0:
                ax.text(0.5, 0.5, "No term", ha="center", va="center",
                        fontsize=7, color="#777777", transform=ax.transAxes)
                continue

            # 子图内部归一化参数
            abs_max, score_shift, score_max_shifted = _panel_norm_params(sub)

            ax.set_ylim(-0.5, n_terms - 0.5)
            sub = sub.reset_index(drop=True)

            for i, (_, r) in enumerate(sub.iterrows()):
                y_center = n_terms - 1 - i

                term_str = str(r["term"])
                abs_delta_val = float(r["abs_imp_delta"])
                score_val = float(r["term_score_mean"])

                # 子图内相对长度
                d_norm = abs_delta_val / (abs_max + 1e-12) if abs_max > 0 else 0.0
                s_norm = (score_val + score_shift) / (score_max_shifted + 1e-12) if score_max_shifted > 0 else 0.0

                d_w = MAX_BAR_WIDTH * float(np.clip(d_norm, 0.0, 1.0))
                s_w = MAX_BAR_WIDTH * float(np.clip(s_norm, 0.0, 1.0))

                # term 标签加白底，彻底避免被柱子干扰
                ax.text(
                    TERM_X, y_center, term_str,
                    fontsize=7, ha="left", va="center",
                    bbox=dict(facecolor="white", edgecolor="none", pad=0.8)
                )

                y_top = y_center + OFFSET
                y_bottom = y_center - OFFSET

                if d_w > 0:
                    ax.barh(y_top, d_w, left=BAR_LEFT, height=BAR_H,
                            color=COLOR_DELTA, edgecolor="white", linewidth=0.5)
                if s_w > 0:
                    ax.barh(y_bottom, s_w, left=BAR_LEFT, height=BAR_H,
                            color=COLOR_SCORE, edgecolor="white", linewidth=0.5)

                # 数值标签只显示数值，贴在柱子右侧
                ax.text(
                    BAR_LEFT + d_w + VALUE_PAD, y_top,
                    f"{abs_delta_val:.2f}",
                    fontsize=6, ha="left", va="center",
                    clip_on=False
                )
                ax.text(
                    BAR_LEFT + s_w + VALUE_PAD, y_bottom,
                    f"{score_val:.2f}",
                    fontsize=6, ha="left", va="center",
                    clip_on=False
                )

    # 标题改成 Type 1 2 3
    for col_idx, z in enumerate(z_list):
        axes[0, col_idx].set_title(f"Type {z}", fontsize=11, pad=6)

    # 行标签
    fig.canvas.draw()
    for row_idx, label in enumerate(["unigram", "multiword"]):
        pos = axes[row_idx, 0].get_position()
        fig.text(0.015, (pos.y0 + pos.y1) / 2.0, label, rotation=90,
                 va="center", ha="center", fontsize=11)

    # 图例只保留在底部，不再在数值标签里重复写
    legend_handles = [
        Patch(facecolor=COLOR_DELTA, label="Risk Variation Index"),
        Patch(facecolor=COLOR_SCORE, label="Tone Variation Index"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 0.02), fontsize=10)

    plt.tight_layout(rect=(0.04, 0.06, 0.99, 0.95))
    fig.savefig(GROUP_FIG_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(GROUP_FIG_PDF, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[save] {GROUP_FIG_PNG}")
    logger.info(f"[save] {GROUP_FIG_PDF}")


# -----------------------------
# 功能 2：词云（按你要求先不动）
# -----------------------------
def plot_impdelta_wordclouds(df_raw: pd.DataFrame) -> None:
    if WordCloud is None:
        logger.error("wordcloud 包未安装，请先 pip install wordcloud")
        return

    for c in ["term", "imp_delta", "risk_type_z"]:
        if c not in df_raw.columns:
            logger.error(f"WordCloud needs column '{c}', but not found.")
            return

    df = df_raw.copy()
    df["term"] = df["term"].astype(str)

    freq_pos, type_pos = _build_wordcloud_freq_and_type(df, sign="pos", topn=WC_TOPN_PER_SIGN)
    freq_neg, type_neg = _build_wordcloud_freq_and_type(df, sign="neg", topn=WC_TOPN_PER_SIGN)

    if not freq_pos and not freq_neg:
        logger.warning("Wordcloud freqs empty, skip.")
        return

    def _boost_top(freq: Dict[str, float]) -> Dict[str, float]:
        if not freq:
            return freq
        items = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        if len(items) >= 1:
            freq[items[0][0]] = items[0][1] * 3.5
        if len(items) >= 2:
            freq[items[1][0]] = items[1][1] * 2.0
        return freq

    freq_pos = _boost_top(freq_pos)
    freq_neg = _boost_top(freq_neg)

    mask = _make_circle_mask(WC_SIZE)
    font_path = _get_default_font_path() or None

    wc = WordCloud(
        width=WC_SIZE, height=WC_SIZE,
        background_color="white",
        mask=mask,
        margin=WC_MARGIN,
        max_words=WC_MAX_WORDS,
        min_font_size=WC_MIN_FONT,
        max_font_size=WC_MAX_FONT,
        prefer_horizontal=1.0,
        random_state=WC_RANDOM_STATE,
        collocations=False,
        scale=1,
        repeat=True,
        font_path=font_path,
        relative_scaling=0,
    )

    def _color_func_factory(type_map: Dict[str, str]):
        def _color_func(word, font_size, position, orientation, random_state=None, **kwargs):
            z = _norm_risk_type(type_map.get(word, "NA"))
            return TYPE_COLOR.get(z, TYPE_FALLBACK)
        return _color_func

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.6), dpi=240)

    ax = axes[0]
    ax.axis("off")
    ax.set_title(WC_TITLE_POS, fontsize=14, pad=10)
    if freq_pos:
        img = wc.generate_from_frequencies(freq_pos).recolor(color_func=_color_func_factory(type_pos))
        ax.imshow(img, interpolation="bilinear")

    ax = axes[1]
    ax.axis("off")
    ax.set_title(WC_TITLE_NEG, fontsize=14, pad=10)
    if freq_neg:
        img = wc.generate_from_frequencies(freq_neg).recolor(color_func=_color_func_factory(type_neg))
        ax.imshow(img, interpolation="bilinear")

    type_handles = [
        Patch(facecolor=TYPE_COLOR["1"], label="Type 1"),
        Patch(facecolor=TYPE_COLOR["2"], label="Type 2"),
        Patch(facecolor=TYPE_COLOR["3"], label="Type 3"),
    ]
    fig.legend(handles=type_handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 0.01), fontsize=10)

    plt.tight_layout(rect=(0.0, 0.05, 1.0, 1.0))
    fig.savefig(WC_FIG_PNG, bbox_inches="tight")
    fig.savefig(WC_FIG_PDF, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[save] {WC_FIG_PNG}")
    logger.info(f"[save] {WC_FIG_PDF}")


# -----------------------------
# main
# -----------------------------
def main(top_k: int = 10) -> None:
    if not os.path.isfile(RES_TABLE_CSV):
        logger.error(f"File not found: {RES_TABLE_CSV}")
        return

    df_raw = pd.read_csv(RES_TABLE_CSV)

    group_df = build_group_table(top_k=top_k)
    if not group_df.empty:
        plot_group_grid(group_df, top_k=top_k)

    plot_impdelta_wordclouds(df_raw)


if __name__ == "__main__":
    main(top_k=10)
