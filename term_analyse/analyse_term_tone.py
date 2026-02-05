# analyse_term_tone.py
# -*- coding: utf-8 -*-
"""
基于 Step 5.2 输出，对 term 语气跨期变化做表格和破产时间差分析。

功能：
1）读取 5.2 输出 recurring_terms_with_significant_tone_shift.csv；
2）读取真实破产时间 bankrupt  all.csv，并对公司名做多级规范化匹配；
3）在 5.2 表中加入 bankrupt_date 列，并在控制台详细输出匹配情况；
4）筛选 is_significant == True 的公司–term，计算：
    - early / late 高峰时间与破产日期的时间差（天数）；
    - early / late 阶段的主导成因、主讲人角色、重复出现次数；
5）输出结果表到：
    /home/ubuntu/project/term_analyse/output/step_5_2_term_tone_shift/table/res_table.csv

多级匹配策略（从严到宽）：
A. company_norm 精确匹配
B. company_key_sorted（二级鲁棒 key，处理法律后缀 + 常见缩写 + 截断 + 短语缩写 + 商标符号）
C. token-stem Jaccard fuzzy fallback
"""

import os
import re
import logging
from typing import Tuple, Dict, List, Set, Optional

import pandas as pd


# -----------------------------
# 路径配置
# -----------------------------

PROJ_ROOT = r"D:\UOW\risk idenfitication\code\risk-identification"

TERM_5_2_CSV = os.path.join(
    PROJ_ROOT,
    "term_analyse",
    "output",
    "step_5_2_term_tone_shift",
    "recurring_terms_with_significant_tone_shift.csv",
)

BANKRUPT_CSV = os.path.join(
    PROJ_ROOT,
    "term_analyse",
    "input",
    "bankrupt  all.csv",
)

OUT_TABLE_DIR = os.path.join(
    PROJ_ROOT,
    "term_analyse",
    "output",
    "step_5_2_term_tone_shift",
    "table",
)
os.makedirs(OUT_TABLE_DIR, exist_ok=True)

OUT_TABLE_CSV = os.path.join(OUT_TABLE_DIR, "res_table.csv")
OUT_UNMATCHED_CSV = os.path.join(OUT_TABLE_DIR, "unmatched_companies.csv")
OUT_MATCH_DEBUG_CSV = os.path.join(OUT_TABLE_DIR, "match_debug.csv")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("analyse_term_tone")


# -----------------------------
# 1) 公司名规范化与多级匹配 key
# -----------------------------

# 可直接删掉的法律/结构后缀
LEGAL_DROP_TOKENS = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY",
    "LTD", "LIMITED", "PLC", "LP", "LLC", "SA", "SAB",
    "BV", "NV", "AG", "GMBH", "PTY",
    # 关键补足：Holding 类
    "HOLDING", "HOLDINGS",
}

# 历史/噪音标记
NOISE_TOKENS = {
    "OLD", "NEW", "FORMER", "FKA", "UNKNOWN",
}

# 常见缩写/截断 → 统一语义 token
ABBREV_CANON = {
    # communications
    "COMM": "COMMUNICATIONS",
    "COMMS": "COMMUNICATIONS",
    "COMMUNICATION": "COMMUNICATIONS",
    "COMMUNICATIONS": "COMMUNICATIONS",

    # holdings
    "HLDG": "HOLDINGS",
    "HLDGS": "HOLDINGS",
    "HOLDING": "HOLDINGS",
    "HOLDINGS": "HOLDINGS",

    # services
    "SVC": "SERVICES",
    "SVCS": "SERVICES",
    "SERVICE": "SERVICES",
    "SERVICES": "SERVICES",

    # technologies
    "TECH": "TECHNOLOGIES",
    "TECHNO": "TECHNOLOGIES",
    "TECHNOLO": "TECHNOLOGIES",
    "TECHNOLOGI": "TECHNOLOGIES",
    "TECHNOLOGY": "TECHNOLOGIES",
    "TECHNOLOGIES": "TECHNOLOGIES",

    # international
    "INTL": "INTERNATIONAL",
    "INTERNATL": "INTERNATIONAL",
    "INTERNATIONAL": "INTERNATIONAL",

    # group
    "GRP": "GROUP",
    "GROUP": "GROUP",

    # 关键补足：Real Estate / Investment(s)
    "RE": "RE",
    "INVS": "INVS",
    "INV": "INVS",
    "INVESTMENT": "INVS",
    "INVESTMENTS": "INVS",
}

# 短语级别缩写（先于 token 化）
PHRASE_REPLACEMENTS = [
    (r"\bREAL\s+ESTATE\b", "RE"),
    (r"\bINVESTMENTS?\b", "INVS"),
]


def normalize_company_name(name: str) -> str:
    """
    规范化为：
        - 清理商标/注册符号 (R)/®/™
        - 去标点
        - 多空格压缩
        - 全大写
        - 短语级替换（REAL ESTATE/INVESTMENT(S)）
    """
    if pd.isna(name):
        return ""

    s = str(name)

    # 1) 清理商标/注册符号
    s = re.sub(r"\(R\)|\(r\)|®|™", "", s)

    # 2) 大写
    s = s.upper()

    # 3) 非字母数字 → 空格
    s = re.sub(r"[^A-Z0-9]+", " ", s)

    # 4) 压缩空格
    s = re.sub(r"\s+", " ", s).strip()

    # 5) 短语级替换
    for pat, rep in PHRASE_REPLACEMENTS:
        s = re.sub(pat, rep, s)

    # 6) 再压一遍空格
    s = re.sub(r"\s+", " ", s).strip()

    return s


def _tokenize_norm(name_norm: str) -> List[str]:
    return name_norm.split() if name_norm else []


def canonicalize_tokens(tokens: List[str]) -> List[str]:
    """
    1) 统一缩写/截断
    2) 删除法律后缀 + 噪音 token
    3) 删除单字符 token
    """
    out = []
    for t in tokens:
        t0 = t.strip().upper()
        if not t0:
            continue
        if len(t0) == 1:
            continue

        t1 = ABBREV_CANON.get(t0, t0)

        if t1 in NOISE_TOKENS:
            continue
        if t1 in LEGAL_DROP_TOKENS:
            continue

        out.append(t1)
    return out


def stem_token(t: str, k: int = 6) -> str:
    t = t.strip().upper()
    return t if len(t) <= k else t[:k]


def build_company_key(name: str, stem_k: int = 6) -> str:
    n = normalize_company_name(name)
    toks = canonicalize_tokens(_tokenize_norm(n))
    stems = [stem_token(t, stem_k) for t in toks if t]
    return " ".join(stems).strip()


def build_company_key_sorted(name: str, stem_k: int = 6) -> str:
    key = build_company_key(name, stem_k=stem_k)
    toks = sorted(set([t for t in key.split() if t]))
    return " ".join(toks).strip()


def token_stem_set(name: str, stem_k: int = 6) -> Set[str]:
    key = build_company_key(name, stem_k=stem_k)
    return set([t for t in key.split() if t])


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# -----------------------------
# 2) bankrupt 表字段识别
# -----------------------------

def detect_columns_for_bankrupt(df: pd.DataFrame) -> Tuple[str, str]:
    """
    bankrupt all.csv：
        - 公司名必须优先用 conm
        - 破产日期优先 dldte
    """
    cols = list(df.columns)
    lower_cols = [c.lower() for c in cols]

    company_col = None
    for c in cols:
        if c.lower() == "conm":
            company_col = c
            break
    if company_col is None:
        for c, lc in zip(cols, lower_cols):
            if "conm" in lc or "company" in lc or "name" in lc:
                company_col = c
                break
    if company_col is None:
        company_col = cols[0]

    date_col = None
    for c, lc in zip(cols, lower_cols):
        if lc == "dldte":
            date_col = c
            break
    if date_col is None:
        for c, lc in zip(cols, lower_cols):
            if "bankrupt" in lc or "bankruptcy" in lc:
                date_col = c
                break
    if date_col is None:
        for c, lc in zip(cols, lower_cols):
            if "date" in lc:
                date_col = c
                break
    if date_col is None:
        non_company_cols = [c for c in cols if c != company_col]
        date_col = non_company_cols[0] if non_company_cols else cols[0]

    logger.info(
        f"[bankrupt] Detected company column: '{company_col}', "
        f"bankrupt date column: '{date_col}'"
    )
    return company_col, date_col


# -----------------------------
# 3) 读取 bankrupt 表
# -----------------------------

def load_bankrupt_table(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Bankrupt CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Bankrupt CSV is empty.")

    logger.info(f"[bankrupt] Raw shape: {df.shape}")
    logger.info(f"[bankrupt] Columns: {list(df.columns)}")

    company_col, date_col = detect_columns_for_bankrupt(df)

    df["company_raw"] = df[company_col].astype(str)
    df["company_norm"] = df["company_raw"].map(normalize_company_name)
    df["bankrupt_date"] = pd.to_datetime(df[date_col], errors="coerce")

    df["company_key"] = df["company_raw"].map(build_company_key)
    df["company_key_sorted"] = df["company_raw"].map(build_company_key_sorted)

    # 去重：同公司多行保留最早日期
    df = (
        df.sort_values(["company_norm", "bankrupt_date"])
        .drop_duplicates("company_norm")
        .loc[:, ["company_raw", "company_norm", "company_key", "company_key_sorted", "bankrupt_date"]]
    )

    logger.info("[bankrupt] Unique companies (normalized): %d", df["company_norm"].nunique())
    logger.info("[bankrupt] Sample rows:\n%s", df.head(5).to_string(index=False))
    return df


# -----------------------------
# 4) 读取 5.2 表
# -----------------------------

def load_term_5_2_table(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"5.2 CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("5.2 CSV is empty.")

    logger.info(f"[5.2] Raw shape: {df.shape}")
    logger.info(f"[5.2] Columns: {list(df.columns)}")

    required_cols = [
        "company", "term",
        "dominant_cause_early", "dominant_cause_late",
        "main_speaker_role_early", "main_speaker_role_late",
        "early_n_sent", "late_n_sent",
        "time_early", "time_late",
        "is_significant",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise KeyError(f"Column '{c}' not found in 5.2 CSV.")

    df["company_norm"] = df["company"].map(normalize_company_name)
    df["company_key"] = df["company"].map(build_company_key)
    df["company_key_sorted"] = df["company"].map(build_company_key_sorted)

    df["time_early_dt"] = pd.to_datetime(df["time_early"], errors="coerce")
    df["time_late_dt"] = pd.to_datetime(df["time_late"], errors="coerce")

    logger.info(
        "[5.2] Unique companies (raw): %d, (normalized): %d",
        df["company"].nunique(),
        df["company_norm"].nunique(),
    )
    logger.info("[5.2] Sample rows:\n%s", df.head(5).to_string(index=False))
    return df


# -----------------------------
# 5) 多级匹配核心（修复版）
# -----------------------------

def attach_bankrupt_date(term_df: pd.DataFrame, bank_df: pd.DataFrame) -> pd.DataFrame:
    """
    三层匹配：
      1) company_norm 精确
      2) company_key_sorted 等值（使用聚合后的 bank_key_map，避免列名冲突）
      3) Jaccard fuzzy fallback
    """

    logger.info("----- Matching companies between 5.2 table and bankrupt table -----")
    logger.info("[match] #companies in 5.2 (normalized): %d", term_df["company_norm"].nunique())
    logger.info("[match] #companies in bankrupt (normalized): %d", bank_df["company_norm"].nunique())
    logger.info("[match] #rows total in 5.2: %d", len(term_df))

    df = term_df.copy()

    # ---- Step1: company_norm 精确匹配 ----
    exact = df.merge(
        bank_df[["company_norm", "bankrupt_date"]],
        on="company_norm",
        how="left",
    )
    exact["match_level"] = exact["bankrupt_date"].notna().map(lambda x: "exact_norm" if x else "")
    logger.info("[match] Step1 exact_norm matched rows: %d", exact["bankrupt_date"].notna().sum())

    # ---- Step2: company_key_sorted 等值匹配 ----
    need_lvl2 = exact["bankrupt_date"].isna()

    if need_lvl2.any():
        # 关键修复：先对 bankrupt 按 key_sorted 聚合，避免同名列冲突与重复 key
        bank_key_map = (
            bank_df.dropna(subset=["company_key_sorted"])
            .groupby("company_key_sorted", as_index=False)["bankrupt_date"]
            .min()
            .rename(columns={"bankrupt_date": "bankrupt_date_key"})
        )

        lvl2_left = exact.loc[need_lvl2, ["company_key_sorted"]].copy()

        lvl2_hit = lvl2_left.merge(
            bank_key_map,
            on="company_key_sorted",
            how="left",
        )

        exact.loc[need_lvl2, "bankrupt_date"] = lvl2_hit["bankrupt_date_key"].values
        exact.loc[need_lvl2, "match_level"] = [
            "key_sorted" if pd.notna(v) else "" for v in lvl2_hit["bankrupt_date_key"].values
        ]

        logger.info("[match] Step2 key_sorted matched rows: %d", exact["match_level"].eq("key_sorted").sum())
    else:
        logger.info("[match] Step2 skipped (all matched by step1)")

    # ---- Step3: fuzzy fallback（公司级映射） ----
    unmatched_mask = exact["bankrupt_date"].isna()
    unmatched_companies = (
        exact.loc[unmatched_mask, ["company", "company_norm"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    logger.info("[match] Companies unmatched after step2: %d", len(unmatched_companies))

    exact["matched_bank_name"] = ""
    exact["match_score"] = 1.0

    if len(unmatched_companies) > 0:
        bank_df2 = bank_df.copy()
        bank_df2["stem_set"] = bank_df2["company_raw"].map(token_stem_set)
        bank_df2["first_stem"] = bank_df2["company_raw"].map(
            lambda x: next(iter(sorted(token_stem_set(x))), "")
        )

        bank_reset = bank_df2.reset_index(drop=True)

        inv_index: Dict[str, List[int]] = {}
        for idx, r in bank_reset.iterrows():
            fs = r.get("first_stem", "")
            if fs:
                inv_index.setdefault(fs, []).append(idx)

        mapping: Dict[str, Tuple[Optional[pd.Timestamp], str, float, str]] = {}

        for _, row in unmatched_companies.iterrows():
            comp_raw = row["company"]
            comp_norm = row["company_norm"]
            sset = token_stem_set(comp_raw)

            if not sset:
                mapping[comp_norm] = (pd.NaT, "", 0.0, "unmatched")
                continue

            first = next(iter(sorted(sset)), "")
            candidates_idx = inv_index.get(first, [])

            best_score = 0.0
            best_date = pd.NaT
            best_name = ""

            search_pool = candidates_idx if candidates_idx else list(range(len(bank_reset)))

            for bi in search_pool:
                b_row = bank_reset.loc[bi]
                score = jaccard(sset, b_row["stem_set"])
                if score > best_score:
                    best_score = score
                    best_date = b_row["bankrupt_date"]
                    best_name = b_row["company_raw"]

            if best_score >= 0.60 and pd.notna(best_date):
                mapping[comp_norm] = (best_date, best_name, best_score, "fuzzy_jaccard")
            else:
                mapping[comp_norm] = (pd.NaT, best_name, best_score, "unmatched")

        def _fill_row(r):
            if pd.notna(r["bankrupt_date"]):
                return r["bankrupt_date"], r["match_level"], "", 1.0
            cn = r["company_norm"]
            bd, bname, score, lvl = mapping.get(cn, (pd.NaT, "", 0.0, "unmatched"))
            if pd.notna(bd):
                return bd, lvl, bname, score
            return pd.NaT, "", bname, score

        filled = exact.apply(_fill_row, axis=1, result_type="expand")
        filled.columns = ["bankrupt_date_filled", "match_level_filled", "matched_bank_name_filled", "match_score_filled"]

        exact["bankrupt_date"] = filled["bankrupt_date_filled"]
        exact["match_level"] = exact["match_level"].where(exact["match_level"] != "", filled["match_level_filled"])
        exact["matched_bank_name"] = filled["matched_bank_name_filled"]
        exact["match_score"] = filled["match_score_filled"]

        logger.info("[match] Step3 fuzzy_jaccard matched rows: %d", exact["match_level"].eq("fuzzy_jaccard").sum())

    # ---- 汇总 ----
    matched_mask = exact["bankrupt_date"].notna()
    matched_company_norm = exact.loc[matched_mask, "company_norm"].dropna().unique()
    unmatched_company_norm = exact.loc[~matched_mask, "company_norm"].dropna().unique()

    logger.info("[match] #rows with matched bankrupt_date (final): %d", matched_mask.sum())
    logger.info("[match] #companies matched (final): %d", len(matched_company_norm))
    logger.info("[match] #companies unmatched (final): %d", len(unmatched_company_norm))

    # debug 输出
    debug_cols = [
        "company", "company_norm", "company_key_sorted",
        "bankrupt_date", "match_level", "matched_bank_name", "match_score"
    ]
    dbg = exact[debug_cols].drop_duplicates().sort_values(["match_level", "company_norm"])
    logger.info("[match] Saving match debug to CSV: %s", OUT_MATCH_DEBUG_CSV)
    dbg.to_csv(OUT_MATCH_DEBUG_CSV, index=False)

    # 仍未匹配公司
    if len(unmatched_company_norm) > 0:
        logger.warning("[match] Example unmatched normalized company names (up to 30):")
        for cn in list(unmatched_company_norm)[:30]:
            raw_names = (
                exact.loc[exact["company_norm"] == cn, "company"]
                .dropna().unique()
            )
            raw_example = raw_names[0] if len(raw_names) > 0 else ""
            logger.warning("    company='%s'  ->  company_norm='%s'", raw_example, cn)

        unmatched_df = (
            exact.loc[~matched_mask, ["company", "company_norm"]]
            .drop_duplicates()
            .sort_values(["company_norm", "company"])
        )
        logger.warning("[match] Saving unmatched companies to CSV: %s", OUT_UNMATCHED_CSV)
        unmatched_df.to_csv(OUT_UNMATCHED_CSV, index=False)

    return exact


# -----------------------------
# 6) 结果表
# -----------------------------

def build_result_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    从合并了 bankrupt_date 的 term_df 里：
    - 先筛选 is_significant == True；
    - 再丢弃 bankrupt_date 仍为空（当前规则下没匹配到）的行；
    - 计算 early / late 距破产天数；
    - 生成最终输出表。
    """
    sig_df = df[df["is_significant"] == True].copy()

    logger.info(
        "[result] Total rows in 5.2 table: %d, is_significant=True rows: %d",
        len(df),
        len(sig_df),
    )

    # 关键新增：不输出仍未匹配到破产日期的行
    before_drop = len(sig_df)
    sig_df = sig_df[sig_df["bankrupt_date"].notna()].copy()
    dropped = before_drop - len(sig_df)

    if dropped > 0:
        logger.warning(
            "[result] Dropped %d is_significant rows due to missing bankrupt_date "
            "(unmatched under current rules).",
            dropped,
        )

    if sig_df.empty:
        logger.warning("[result] No rows left after dropping unmatched bankrupt_date.")
        return pd.DataFrame(
            columns=[
                "company",
                "term",
                "dominant_cause_early",
                "days_to_bankrupt_early",
                "main_speaker_role_early",
                "early_n_sent",
                "dominant_cause_late",
                "days_to_bankrupt_late",
                "main_speaker_role_late",
                "late_n_sent",
            ]
        )

    sig_df["days_to_bankrupt_early"] = (
        sig_df["bankrupt_date"] - sig_df["time_early_dt"]
    ).dt.days

    sig_df["days_to_bankrupt_late"] = (
        sig_df["bankrupt_date"] - sig_df["time_late_dt"]
    ).dt.days

    logger.info(
        "[result] days_to_bankrupt_early: min=%s, max=%s",
        sig_df["days_to_bankrupt_early"].min(),
        sig_df["days_to_bankrupt_early"].max(),
    )
    logger.info(
        "[result] days_to_bankrupt_late : min=%s, max=%s",
        sig_df["days_to_bankrupt_late"].min(),
        sig_df["days_to_bankrupt_late"].max(),
    )

    result_cols = [
        "company",
        "term",
        "dominant_cause_early",
        "days_to_bankrupt_early",
        "main_speaker_role_early",
        "early_n_sent",
        "dominant_cause_late",
        "days_to_bankrupt_late",
        "main_speaker_role_late",
        "late_n_sent",
    ]

    result = sig_df[result_cols].copy()

    result = result.sort_values(
        ["company", "days_to_bankrupt_early", "term"],
        ascending=[True, True, True],
    )

    logger.info("[result] Final result table shape: %s", result.shape)
    logger.info("[result] Sample rows:\n%s", result.head(10).to_string(index=False))

    return result


# -----------------------------
# 主入口
# -----------------------------

def main():
    logger.info("===== Analyse term tone & distance to bankruptcy START =====")

    term_df = load_term_5_2_table(TERM_5_2_CSV)
    logger.info("[main] Loaded 5.2 term table: %d rows", len(term_df))

    bank_df = load_bankrupt_table(BANKRUPT_CSV)
    logger.info("[main] Loaded bankrupt table: %d rows", len(bank_df))

    merged_df = attach_bankrupt_date(term_df, bank_df)

    res_table = build_result_table(merged_df)
    logger.info("[main] Result table rows (is_significant=True): %d", len(res_table))

    logger.info("[main] Saving result table to %s", OUT_TABLE_CSV)
    res_table.to_csv(OUT_TABLE_CSV, index=False)

    logger.info("===== Analyse term tone & distance to bankruptcy DONE =====")


if __name__ == "__main__":
    main()
