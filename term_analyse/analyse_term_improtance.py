# analyse_term_improtance.py
# -*- coding: utf-8 -*-
"""
Analyse term importance summary (Step 5.3 输出汇总版 → 生成简表)

【当前版本仅针对 clean_table】

输入：
    /home/ubuntu/project/term_analyse/output/importance/term_importance_summary_clean.csv

输出：
    /home/ubuntu/project/term_analyse/output/importance/table/res_table_clean.csv

过滤规则（稳健保留）：
    imp_early、imp_late、imp_delta 三个指标中至少有一个 ≠ 0
    （当前实现依然采用“全非 0 更稳健”的 & 逻辑）

输出列：
    company,
    risk_type_z,
    term,
    imp_early,
    imp_late,
    imp_delta,
    term_score_mean,
    mode_speaker_early,
    mode_speaker_late,
    mode_dominant_cause_early,
    mode_dominant_cause_late,
    time_early,
    time_late,
    early_n_sent,
    late_n_sent

并在 term_tone_stats.json 基础上新增：
    unique_terms_with_nonzero_imp_delta
    company_term_rows_with_nonzero_imp_delta
"""

import os
import json
import logging
import pandas as pd

# -----------------------------
# 路径配置
# -----------------------------

PROJ_ROOT = r"D:\UOW\risk idenfitication\code\risk-identification"

SUMMARY_CLEAN_CSV = os.path.join(
    PROJ_ROOT,
    "term_analyse",
    "output",
    "importance",
    "term_importance_summary_clean.csv",
)

OUT_TABLE_DIR = os.path.join(
    PROJ_ROOT,
    "term_analyse",
    "output",
    "importance",
    "table",
)
os.makedirs(OUT_TABLE_DIR, exist_ok=True)

OUT_TABLE_CSV = os.path.join(OUT_TABLE_DIR, "res_table_clean.csv")

# ★ 来自 Step 5.2 的 stats JSON（要在此基础上追加 imp_delta 相关统计）
TERM_TONE_STATS_JSON = os.path.join(
    PROJ_ROOT,
    "term_analyse",
    "output",
    "step_5_2_term_tone_shift",
    "term_tone_stats.json",
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("analyse_term_importance_clean")


# -----------------------------
# 主逻辑
# -----------------------------

def main():
    logger.info("===== Analyse term importance CLEAN summary START =====")

    # 1) 读取 clean summary 表
    if not os.path.isfile(SUMMARY_CLEAN_CSV):
        logger.error(f"Clean summary CSV not found: {SUMMARY_CLEAN_CSV}")
        return

    df = pd.read_csv(SUMMARY_CLEAN_CSV)
    logger.info(f"[load] Raw shape: {df.shape}")
    logger.info(f"[load] Columns: {list(df.columns)}")

    required_cols = [
        "company",
        "risk_type_z",
        "term",
        "imp_early",
        "imp_late",
        "imp_delta",
        "term_score_mean",
        "mode_speaker_early",
        "mode_speaker_late",
        "mode_dominant_cause_early",
        "mode_dominant_cause_late",
        "time_early",
        "time_late",
        "early_n_sent",
        "late_n_sent",
    ]
    for c in required_cols:
        if c not in df.columns:
            logger.error(f"Required column '{c}' not found in clean summary CSV.")
            return

    # 2) 把 imp_* 转成数值，NaN 当 0
    for c in ["imp_early", "imp_late", "imp_delta"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # ★ 在过滤之前就统计 "有 imp_delta 的 term 数量"
    mask_imp_delta_nonzero = df["imp_delta"] != 0.0
    unique_terms_with_nonzero_imp_delta = int(
        df.loc[mask_imp_delta_nonzero, "term"].nunique()
    )
    company_term_rows_with_nonzero_imp_delta = int(mask_imp_delta_nonzero.sum())

    before_filter = len(df)

    # 3) 稳健过滤（即便 clean 表理论上已无全 0 行）
    # 当前逻辑：三项都≠0才保留，更保守
    mask_non_zero = (
        (df["imp_early"] != 0.0)
        & (df["imp_late"] != 0.0)
        & (df["imp_delta"] != 0.0)
    )
    df = df[mask_non_zero].copy()

    after_filter = len(df)
    logger.info(
        f"[filter] Rows before filter: {before_filter}, "
        f"rows with all imp_* != 0: {after_filter}"
    )

    # 4) 只保留需要的列（这里已经包含 time_* 和 *_n_sent）
    res_df = df[required_cols].copy()

    # 可选排序：先按公司，再按 imp_delta 从大到小
    res_df = res_df.sort_values(
        by=["company", "imp_delta"],
        ascending=[True, False],
    )

    logger.info(f"[result] Final res_table_clean shape: {res_df.shape}")
    if not res_df.empty:
        logger.info("[result] Sample rows:\n%s", res_df.head(10).to_string(index=False))

    # 5) 输出简表
    logger.info(f"[save] Saving res_table_clean to {OUT_TABLE_CSV}")
    res_df.to_csv(OUT_TABLE_CSV, index=False)

    # 6) ★ 更新 term_tone_stats.json，追加 imp_delta 相关统计
    try:
        if os.path.isfile(TERM_TONE_STATS_JSON):
            try:
                with open(TERM_TONE_STATS_JSON, "r", encoding="utf-8") as f:
                    stats = json.load(f)
            except Exception as e:
                logger.warning(
                    f"Failed to load existing term_tone_stats.json: {e}. "
                    f"Recreating stats dict."
                )
                stats = {}
        else:
            logger.warning(
                "term_tone_stats.json not found. "
                "Creating a new stats dict with imp_delta info only."
            )
            stats = {}

        stats["company_term_rows_with_nonzero_imp_delta"] = company_term_rows_with_nonzero_imp_delta
        # 这里可以顺带存一下 term 数量，虽然之前没用到
        stats["unique_terms_with_nonzero_imp_delta"] = unique_terms_with_nonzero_imp_delta

        with open(TERM_TONE_STATS_JSON, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(
            f"[stats] Updated term_tone_stats.json with imp_delta info: "
            f"unique_terms_with_nonzero_imp_delta={unique_terms_with_nonzero_imp_delta}, "
            f"company_term_rows_with_nonzero_imp_delta={company_term_rows_with_nonzero_imp_delta}"
        )
    except Exception as e:
        logger.error(f"Failed to update term_tone_stats.json with imp_delta stats: {e}")

    logger.info("===== Analyse term importance CLEAN summary DONE =====")


if __name__ == "__main__":
    main()
