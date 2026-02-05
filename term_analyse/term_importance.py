# term_importance.py
# -*- coding: utf-8 -*-
"""
Step 5.3 —— Term-level 风险重要性分析（基于 5.2 + RQ1 surrogate LR）

输入：
    1) 5.2 输出（term 跨期语气显著变化）：
       /home/ubuntu/project/term_analyse/output/step_5_2_term_tone_shift/recurring_terms_with_significant_tone_shift.csv

    2) RQ1 surrogate LR 组件（与 attribution.py 一致）：
       /home/ubuntu/project/predict/output/fijz_logratio.csv
       /home/ubuntu/project/predict/output/coef_multinomial_long.csv
       /home/ubuntu/project/predict/output/preds_multinomial.csv

    3) 真实破产时间：
       /home/ubuntu/project/term_analyse/input/bankrupt  all.csv
       （公司名在 conm，破产日期在 dldte）

核心思路：
    - 使用 RQ1 同一套逻辑回归 surrogate。
    - 将 company 级别 a_mij 换为 term 对应的 early_n_sent / late_n_sent。
    - 对每一条 (company, term)：
        early 使用 speaker_early + dominant_cause_early；
        late  使用 speaker_late  + dominant_cause_late。
      真实风险类型 z 由 source_type 决定。
    - 重要性计算：
        C_logit_early = unit_logit(i1, j1, z) * early_n_sent
        C_logit_late  = unit_logit(i2, j2, z) * late_n_sent
        importance_*  = p_mz * (1 - p_mz) * C_logit_*
        delta         = late - early
    - 接入真实破产时间 bankrupt_date，并计算 days_to_bankrupt_early / late。

输出：
    1) term_importance_detailed_by_company.csv
    2) term_importance_summary.csv
    3) term_importance_summary_clean.csv
"""

import os
import re
import logging
from typing import Tuple, Dict, Any, List, Set, Optional

import numpy as np
import pandas as pd

# -----------------------------
# 路径配置
# -----------------------------

PROJ_ROOT = r"D:\UOW\risk idenfitication\code\risk-identification"

RECURRING_TERMS_PATH = os.path.join(
    PROJ_ROOT,
    "term_analyse",
    "output",
    "step_5_2_term_tone_shift",
    "recurring_terms_with_significant_tone_shift.csv",
)

PREDICT_OUT_ROOT = os.path.join(PROJ_ROOT, "predict", "output")

FIJZ_CSV = os.path.join(PREDICT_OUT_ROOT, "fijz_logratio.csv")
COEF_CSV = os.path.join(PREDICT_OUT_ROOT, "coef_multinomial_long.csv")
PREDS_CSV = os.path.join(PREDICT_OUT_ROOT, "preds_multinomial.csv")

BANKRUPT_CSV = os.path.join(
    PROJ_ROOT,
    "term_analyse",
    "input",
    "bankrupt  all.csv",
)

OUT_ROOT = os.path.join(
    PROJ_ROOT,
    "term_analyse",
    "output",
    "importance",
)
os.makedirs(OUT_ROOT, exist_ok=True)

DETAILED_OUT = os.path.join(OUT_ROOT, "term_importance_detailed_by_company.csv")
SUMMARY_OUT = os.path.join(OUT_ROOT, "term_importance_summary.csv")
SUMMARY_CLEAN_OUT = os.path.join(OUT_ROOT, "term_importance_summary_clean.csv")

UNMATCHED_CSV = os.path.join(OUT_ROOT, "unmatched_companies_from_bankrupt.csv")
MATCH_DEBUG_CSV = os.path.join(OUT_ROOT, "match_debug_from_bankrupt.csv")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("step_5_3_term_importance")

CANON_RISKS = [
    "risk_uncertainty",
    "risk_legal",
    "risk_constraint",
    "risk_external",
]


# -----------------------------
# is_significant 工具
# -----------------------------

def safe_astype_int_bool(series: pd.Series) -> pd.Series:
    """
    将 is_significant 这类列安全地转成 {0,1} 整型。
    """
    if series.dtype == bool:
        return series.astype(int)
    if series.dtype == object:
        s = series.fillna("")
        s = s.astype(str).str.lower().str.strip()
        return s.isin(["1", "true", "yes", "y"]).astype(int)
    return (series.fillna(0).astype(float) != 0.0).astype(int)


# -----------------------------
# 公司名多级匹配规则（与 analyse_term_tone 对齐）
# -----------------------------

LEGAL_DROP_TOKENS = {
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY",
    "LTD", "LIMITED", "PLC", "LP", "LLC", "SA", "SAB",
    "BV", "NV", "AG", "GMBH", "PTY",
    "HOLDING", "HOLDINGS",
}

NOISE_TOKENS = {
    "OLD", "NEW", "FORMER", "FKA", "UNKNOWN",
}

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

    # investment shorthand
    "INVS": "INVS",
    "INV": "INVS",
    "INVESTMENT": "INVS",
    "INVESTMENTS": "INVS",
}

PHRASE_REPLACEMENTS = [
    (r"\bREAL\s+ESTATE\b", "RE"),
    (r"\bINVESTMENTS?\b", "INVS"),
]


def normalize_company_name(name: Any) -> str:
    """
    规范化公司名：
      - 去掉商标/注册符号
      - 大写
      - 非字母数字 -> 空格
      - 短语级替换
      - 压缩空格
    """
    if pd.isna(name):
        return ""

    s = str(name)
    s = re.sub(r"\(R\)|\(r\)|®|™", "", s)

    s = s.upper()
    s = re.sub(r"[^A-Z0-9]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    for pat, rep in PHRASE_REPLACEMENTS:
        s = re.sub(pat, rep, s)

    s = re.sub(r"\s+", " ", s).strip()
    return s


def _tokenize_norm(name_norm: str) -> List[str]:
    return name_norm.split() if name_norm else []


def canonicalize_tokens(tokens: List[str]) -> List[str]:
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


def build_company_key(name: Any, stem_k: int = 6) -> str:
    n = normalize_company_name(name)
    toks = _tokenize_norm(n)
    toks = canonicalize_tokens(toks)
    stems = [stem_token(t, stem_k) for t in toks if t]
    return " ".join(stems).strip()


def build_company_key_sorted(name: Any, stem_k: int = 6) -> str:
    key = build_company_key(name, stem_k=stem_k)
    toks = [t for t in key.split() if t]
    toks = sorted(set(toks))
    return " ".join(toks).strip()


def token_stem_set(name: Any, stem_k: int = 6) -> Set[str]:
    key = build_company_key(name, stem_k=stem_k)
    return set([t for t in key.split() if t])


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


# -----------------------------
# 风险、众数、source_type -> z
# -----------------------------

def normalize_risk_bucket(val: Any) -> str:
    if pd.isna(val):
        return ""
    s = str(val).strip().lower()
    if not s:
        return s

    if s in CANON_RISKS:
        return s

    bare = s
    if bare.startswith("risk_"):
        bare = bare[5:]

    if bare in ["uncertainty", "legal", "constraint", "external"]:
        return f"risk_{bare}"

    return s


def mode_or_unknown(x: pd.Series, default: str = "UNKNOWN") -> str:
    x = x.dropna().astype(str)
    if x.empty:
        return default
    m = x.mode()
    return m.iloc[0] if not m.empty else default


def map_source_type_to_z(series: pd.Series) -> pd.Series:
    s = series.astype(str).fillna("")
    z_vals = []
    for v in s:
        m = re.search(r"(\d+)", v)
        if m:
            try:
                z_vals.append(int(m.group(1)))
            except Exception:
                z_vals.append(np.nan)
        else:
            z_vals.append(np.nan)
    return pd.Series(z_vals, index=series.index, dtype="float")


def _infer_prob_cols(preds: pd.DataFrame):
    cols = preds.columns.tolist()
    strict = [c for c in cols if re.fullmatch(r"prob_z[123]", str(c))]
    if len(strict) == 3:
        return ["prob_z1", "prob_z2", "prob_z3"]

    mapping = {}
    for c in cols:
        s = str(c).lower()
        m = re.search(r"(prob|p)[_ ]*z?([123])$", s)
        if m:
            mapping[int(m.group(2))] = c
    if all(z in mapping for z in [1, 2, 3]):
        return [mapping[1], mapping[2], mapping[3]]
    return None


# -----------------------------
# bankrupt all.csv 读取
# -----------------------------

def detect_columns_for_bankrupt(df: pd.DataFrame) -> Tuple[str, str]:
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
        non_company = [c for c in cols if c != company_col]
        if not non_company:
            raise ValueError("Cannot detect bankrupt date column in bankrupt all.csv")
        date_col = non_company[0]

    logger.info(
        f"[bankrupt] Detected company column: '{company_col}', "
        f"bankrupt date column: '{date_col}'"
    )
    return company_col, date_col


def load_bankrupt_table(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Bankrupt CSV not found: {path}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Bankrupt CSV is empty")

    logger.info(f"[bankrupt] Raw shape: {df.shape}")
    logger.info(f"[bankrupt] Columns: {list(df.columns)}")

    company_col, date_col = detect_columns_for_bankrupt(df)

    df["company_raw"] = df[company_col].astype(str)
    df["company_norm"] = df["company_raw"].map(normalize_company_name)
    df["company_key"] = df["company_raw"].map(build_company_key)
    df["company_key_sorted"] = df["company_raw"].map(build_company_key_sorted)

    df["bankrupt_date"] = pd.to_datetime(df[date_col], errors="coerce")

    df = (
        df.sort_values(["company_norm", "bankrupt_date"])
        .drop_duplicates("company_norm")
        .loc[:, ["company_raw", "company_norm", "company_key", "company_key_sorted", "bankrupt_date"]]
    )

    logger.info("[bankrupt] Unique companies (normalized): %d", df["company_norm"].nunique())
    logger.info("[bankrupt] Sample rows:\n%s", df.head(5).to_string(index=False))
    return df


# -----------------------------
# 5.2 读取与过滤
# -----------------------------

def load_and_filter_recurring_terms(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        logger.error(f"Recurring terms CSV not found: {path}")
        return pd.DataFrame()

    logger.info(f"Loading recurring terms with tone shift from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} rows from recurring_terms_with_significant_tone_shift.csv")

    required_basic_cols = ["term", "company", "source_type"]
    for col in required_basic_cols:
        if col not in df.columns:
            logger.error(
                f"Required column '{col}' not found. Columns = {list(df.columns)}"
            )
            return pd.DataFrame()

    if "is_significant" in df.columns:
        sig = safe_astype_int_bool(df["is_significant"])
        logger.info(f"is_significant column found. Significant rows = {sig.sum()}")
        df = df[sig == 1].copy()
    else:
        logger.warning("Column 'is_significant' not found. Using ALL rows as significant.")

    df = df.dropna(subset=["term", "company", "source_type"]).copy()

    df["company"] = df["company"].astype(str)
    df["company_norm"] = df["company"].map(normalize_company_name)
    df["company_key"] = df["company"].map(build_company_key)
    df["company_key_sorted"] = df["company"].map(build_company_key_sorted)

    # 成因
    if "dominant_cause_early" not in df.columns:
        df["dominant_cause_early"] = np.nan
    if "dominant_cause_late" not in df.columns:
        df["dominant_cause_late"] = np.nan

    df["dominant_cause_early"] = df["dominant_cause_early"].apply(normalize_risk_bucket)
    df["dominant_cause_late"] = df["dominant_cause_late"].apply(normalize_risk_bucket)

    # speaker
    if "main_speaker_role" not in df.columns:
        df["main_speaker_role"] = "UNKNOWN"

    if "main_speaker_role_early" in df.columns:
        df["speaker_early"] = df["main_speaker_role_early"]
    else:
        df["speaker_early"] = df["main_speaker_role"]

    if "main_speaker_role_late" in df.columns:
        df["speaker_late"] = df["main_speaker_role_late"]
    else:
        df["speaker_late"] = df["main_speaker_role"]

    df["speaker_early"] = df["speaker_early"].fillna("UNKNOWN").astype(str).str.strip()
    df["speaker_late"] = df["speaker_late"].fillna("UNKNOWN").astype(str).str.strip()
    df["main_speaker_role"] = df["main_speaker_role"].fillna("UNKNOWN").astype(str).str.strip()

    # early/late 句子数
    if "early_n_sent" not in df.columns:
        df["early_n_sent"] = 0.0
    if "late_n_sent" not in df.columns:
        df["late_n_sent"] = 0.0

    df["early_n_sent"] = pd.to_numeric(df["early_n_sent"], errors="coerce").fillna(0.0)
    df["late_n_sent"] = pd.to_numeric(df["late_n_sent"], errors="coerce").fillna(0.0)

    # 时间
    df["time_early_dt"] = (
        pd.to_datetime(df["time_early"], errors="coerce") if "time_early" in df.columns else pd.NaT
    )
    df["time_late_dt"] = (
        pd.to_datetime(df["time_late"], errors="coerce") if "time_late" in df.columns else pd.NaT
    )

    logger.info(f"Remaining significant rows = {len(df)}")
    return df


# -----------------------------
# 5.3 使用的多级 bankrupt_date 匹配
# -----------------------------

def attach_bankrupt_date(term_df: pd.DataFrame, bank_df: pd.DataFrame) -> pd.DataFrame:
    """
    多级匹配：
      1) company_norm 精确
      2) company_key_sorted 等值
      3) token-stem Jaccard fallback（公司级映射）
    """
    logger.info("----- Matching companies between 5.2 table and bankrupt table (5.3 rules) -----")
    logger.info("[match] #companies in terms (norm): %d", term_df["company_norm"].nunique())
    logger.info("[match] #companies in bankrupt (norm): %d", bank_df["company_norm"].nunique())
    logger.info("[match] #rows in terms: %d", len(term_df))

    df = term_df.copy()

    # 1) exact norm
    exact = df.merge(
        bank_df[["company_norm", "bankrupt_date"]],
        on="company_norm",
        how="left",
    )
    exact["match_level"] = exact["bankrupt_date"].notna().map(lambda x: "exact_norm" if x else "")
    logger.info("[match] Step1 exact_norm matched rows: %d", exact["bankrupt_date"].notna().sum())

    # 2) key_sorted
    need_lvl2 = exact["bankrupt_date"].isna()
    if need_lvl2.any():
        lvl2_left = exact.loc[need_lvl2, :].copy()

        lvl2 = lvl2_left.merge(
            bank_df[["company_key_sorted", "bankrupt_date"]].dropna(),
            on="company_key_sorted",
            how="left",
            suffixes=("_x", "_y"),
        )

        # 定位合并后的 bankrupt_date 列
        lvl2_date_col = None
        if "bankrupt_date_y" in lvl2.columns:
            lvl2_date_col = "bankrupt_date_y"
        elif "bankrupt_date" in lvl2.columns:
            lvl2_date_col = "bankrupt_date"
        else:
            cand = [c for c in lvl2.columns if c.startswith("bankrupt_date")]
            cand_y = [c for c in cand if c.endswith("_y")]
            if cand_y:
                lvl2_date_col = cand_y[0]
            elif cand:
                lvl2_date_col = cand[-1]

        if lvl2_date_col:
            vals = lvl2[lvl2_date_col].values
            exact.loc[need_lvl2, "bankrupt_date"] = vals
            exact.loc[need_lvl2, "match_level"] = [
                "key_sorted" if pd.notna(v) else "" for v in vals
            ]
            logger.info("[match] Step2 key_sorted matched rows: %d", exact["match_level"].eq("key_sorted").sum())
        else:
            logger.warning("[match] Step2 key_sorted failed to locate merged bankrupt_date column.")
    else:
        logger.info("[match] Step2 skipped")

    # 3) fuzzy fallback
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
        bank_work = bank_df.copy()
        bank_work["stem_set"] = bank_work["company_raw"].map(token_stem_set)
        bank_reset = bank_work.reset_index(drop=True)

        mapping: Dict[str, Tuple[Optional[pd.Timestamp], str, float, str]] = {}

        for _, row in unmatched_companies.iterrows():
            comp_raw = row["company"]
            comp_norm = row["company_norm"]
            sset = token_stem_set(comp_raw)

            best_score = 0.0
            best_date = pd.NaT
            best_name = ""

            for bi in range(len(bank_reset)):
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
            comp_norm = r["company_norm"]
            bd, bname, score, lvl = mapping.get(comp_norm, (pd.NaT, "", 0.0, "unmatched"))
            if pd.notna(bd):
                return bd, lvl, bname, score
            return pd.NaT, "", bname, score

        filled = exact.apply(_fill_row, axis=1, result_type="expand")
        filled.columns = [
            "bankrupt_date_filled",
            "match_level_filled",
            "matched_bank_name_filled",
            "match_score_filled",
        ]

        exact["bankrupt_date"] = filled["bankrupt_date_filled"]
        exact["match_level"] = exact["match_level"].where(
            exact["match_level"] != "", filled["match_level_filled"]
        )
        exact["matched_bank_name"] = filled["matched_bank_name_filled"]
        exact["match_score"] = filled["match_score_filled"]

        logger.info("[match] Step3 fuzzy_jaccard matched rows: %d", exact["match_level"].eq("fuzzy_jaccard").sum())

    # 汇总与输出 debug
    matched_mask = exact["bankrupt_date"].notna()
    unmatched_company_norm = exact.loc[~matched_mask, "company_norm"].dropna().unique()

    logger.info("[match] #rows with matched bankrupt_date (final): %d", matched_mask.sum())
    logger.info("[match] #companies unmatched (final): %d", len(unmatched_company_norm))

    debug_cols = [
        "company", "company_norm", "company_key_sorted",
        "bankrupt_date", "match_level", "matched_bank_name", "match_score",
    ]
    dbg = exact[debug_cols].drop_duplicates().sort_values(["match_level", "company_norm"])
    logger.info("[match] Saving match debug to CSV: %s", MATCH_DEBUG_CSV)
    dbg.to_csv(MATCH_DEBUG_CSV, index=False)

    if len(unmatched_company_norm) > 0:
        unmatched_df = (
            exact.loc[~matched_mask, ["company", "company_norm"]]
            .drop_duplicates()
            .sort_values(["company_norm", "company"])
        )
        logger.warning("[match] Saving unmatched companies to CSV: %s", UNMATCHED_CSV)
        unmatched_df.to_csv(UNMATCHED_CSV, index=False)

    return exact


# -----------------------------
# 加载 RQ1 surrogate LR 组件
# -----------------------------

def load_fijz_and_pivot(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        logger.error(f"fijz_logratio CSV not found: {path}")
        return pd.DataFrame()

    f = pd.read_csv(path)
    for c in ["role", "risk_bucket", "z_label", "f_ijz"]:
        if c not in f.columns:
            raise ValueError(f"[fijz_logratio.csv] missing column: {c}. Got: {list(f.columns)}")

    f["role"] = f["role"].astype(str).str.strip()
    f["risk_bucket"] = f["risk_bucket"].astype(str).str.strip().apply(normalize_risk_bucket)
    f["z_label"] = pd.to_numeric(f["z_label"], errors="coerce").astype("Int64")
    f["f_ijz"] = pd.to_numeric(f["f_ijz"], errors="coerce").fillna(0.0)

    f = f[f["z_label"].notna()].copy()
    f["z_label"] = f["z_label"].astype(int)

    f_pvt = f.pivot_table(
        index=["role", "risk_bucket"],
        columns="z_label",
        values="f_ijz",
        aggfunc="first",
    ).fillna(0.0)

    for z in [1, 2, 3]:
        if z not in f_pvt.columns:
            f_pvt[z] = 0.0
    f_pvt = f_pvt[[1, 2, 3]]

    logger.info(f"Built f_ijz pivot with shape={f_pvt.shape}")
    return f_pvt


def build_coef_maps(coef_long: pd.DataFrame):
    beta_map: Dict[int, Dict[str, float]] = {}
    gamma_map: Dict[int, Dict[str, float]] = {}

    for z, g in coef_long.groupby("z_label"):
        beta, gamma = {}, {}
        for _, r in g.iterrows():
            feat = str(r["feature"]).strip()
            val = float(r["coef"])

            if feat.lower() in ["intercept", "(intercept)", "bias", "const"]:
                continue

            if feat.startswith("prior_z"):
                gamma[feat] = val
            else:
                beta[feat] = val

        beta_map[int(z)] = beta
        gamma_map[int(z)] = gamma

    return beta_map, gamma_map


def load_coef_long(path: str):
    if not os.path.isfile(path):
        logger.error(f"coef_multinomial_long CSV not found: {path}")
        return {}, {}

    coef = pd.read_csv(path)

    coef_cols = {c.lower(): c for c in coef.columns}
    c_z = coef_cols.get("z_label") or coef_cols.get("class") or coef_cols.get("z")
    c_f = coef_cols.get("feature") or coef_cols.get("feat")
    c_b = coef_cols.get("coef") or coef_cols.get("beta") or coef_cols.get("weight")

    if not (c_z and c_f and c_b):
        raise ValueError(
            f"[coef_multinomial_long.csv] need z_label/feature/coef cols. Got: {list(coef.columns)}"
        )

    coef = coef[[c_z, c_f, c_b]].rename(
        columns={c_z: "z_label", c_f: "feature", c_b: "coef"}
    )
    coef["z_label"] = pd.to_numeric(coef["z_label"], errors="coerce").astype("Int64")
    coef["feature"] = coef["feature"].astype(str)
    coef["coef"] = pd.to_numeric(coef["coef"], errors="coerce").fillna(0.0)

    coef = coef[coef["z_label"].notna()].copy()
    coef["z_label"] = coef["z_label"].astype(int)

    beta_map, gamma_map = build_coef_maps(coef)
    logger.info("Built beta_map / gamma_map from coef_multinomial_long.csv")
    return beta_map, gamma_map


def load_company_probs(path: str) -> pd.DataFrame:
    """
    读取 preds_multinomial.csv。
    返回：company_raw, company_norm, z_label, pmz
    """
    if not os.path.isfile(path):
        logger.error(f"preds_multinomial CSV not found: {path}")
        return pd.DataFrame()

    preds = pd.read_csv(path)
    if "company_id" not in preds.columns:
        raise ValueError(
            f"[preds_multinomial.csv] missing 'company_id'. Got: {list(preds.columns)}"
        )

    prob_cols = _infer_prob_cols(preds)
    if prob_cols is None:
        raise ValueError(
            f"[preds_multinomial.csv] cannot find prob cols like prob_z1/prob_z2/prob_z3. "
            f"Got columns: {list(preds.columns)}"
        )

    preds["company_raw"] = preds["company_id"].astype(str)
    preds["company_norm"] = preds["company_raw"].map(normalize_company_name)

    preds = preds.rename(
        columns={prob_cols[0]: "prob_z1", prob_cols[1]: "prob_z2", prob_cols[2]: "prob_z3"}
    )
    for c in ["prob_z1", "prob_z2", "prob_z3"]:
        preds[c] = pd.to_numeric(preds[c], errors="coerce").fillna(0.0)

    probs = preds[["company_raw", "company_norm", "prob_z1", "prob_z2", "prob_z3"]].copy()
    probs = probs.melt(
        id_vars=["company_raw", "company_norm"],
        var_name="pcol",
        value_name="pmz"
    )
    probs["z_label"] = probs["pcol"].str.extract(r"(\d)").astype(int)
    probs = probs[["company_raw", "company_norm", "z_label", "pmz"]]

    logger.info(f"Built company-level prob table with shape={probs.shape}")
    return probs


def load_predict_components():
    f_pvt = load_fijz_and_pivot(FIJZ_CSV)
    beta_map, gamma_map = load_coef_long(COEF_CSV)
    company_probs = load_company_probs(PREDS_CSV)

    if f_pvt.empty or not beta_map or company_probs.empty:
        logger.error("Failed to load one or more RQ1 components.")
    return f_pvt, beta_map, gamma_map, company_probs


# -----------------------------
# unit_logit 预计算
# -----------------------------

def build_unit_logit_table(
    f_pvt: pd.DataFrame,
    beta_map: Dict[int, Dict[str, float]],
    gamma_map: Dict[int, Dict[str, float]],
) -> pd.DataFrame:
    if f_pvt.empty:
        return pd.DataFrame(columns=["role", "risk_bucket", "z_label", "unit_logit"])

    pairs = set((r, rb) for (r, rb) in f_pvt.index)

    for z, beta in beta_map.items():
        for feat in beta.keys():
            if "|" not in feat:
                continue
            role, risk = feat.split("|", 1)
            risk_norm = normalize_risk_bucket(risk)
            pairs.add((role, risk_norm))

    rows = []
    for role, risk in pairs:
        if (role, risk) in f_pvt.index:
            f1 = float(f_pvt.loc[(role, risk), 1])
            f2 = float(f_pvt.loc[(role, risk), 2])
            f3 = float(f_pvt.loc[(role, risk), 3])
        else:
            f1 = f2 = f3 = 0.0

        for z in (1, 2, 3):
            beta_z = beta_map.get(z, {})
            gamma_z = gamma_map.get(z, {})

            feat = f"{role}|{risk}"
            b = float(beta_z.get(feat, 0.0))

            v1 = float(gamma_z.get("prior_z1", 0.0)) * f1
            v2 = float(gamma_z.get("prior_z2", 0.0)) * f2
            v3 = float(gamma_z.get("prior_z3", 0.0)) * f3

            unit_logit = b + v1 + v2 + v3

            rows.append(
                {"role": role, "risk_bucket": risk, "z_label": int(z), "unit_logit": float(unit_logit)}
            )

    unit_df = pd.DataFrame(rows)
    logger.info(f"Built unit_logit table with shape={unit_df.shape}")
    return unit_df


# -----------------------------
# 重要性计算
# -----------------------------

def compute_term_importance(
    df_terms: pd.DataFrame,
    unit_logit_df: pd.DataFrame,
    company_probs: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    if df_terms.empty:
        logger.error("df_terms is empty.")
        return pd.DataFrame(), pd.DataFrame()
    if unit_logit_df.empty or company_probs.empty:
        logger.error("unit_logit or company_probs is empty.")
        return pd.DataFrame(), pd.DataFrame()

    df = df_terms.copy().reset_index(drop=True)

    # z_label_true
    df["z_label_true"] = map_source_type_to_z(df["source_type"])
    df["z_label_true"] = df["z_label_true"].fillna(-1).astype(int)

    before_z = len(df)
    df = df[df["z_label_true"] > 0].copy()
    if len(df) < before_z:
        logger.info(f"Dropped {before_z - len(df)} rows with invalid z_label_true.")

    # bankrupt days
    if "bankrupt_date" in df.columns:
        df["days_to_bankrupt_early"] = (df["bankrupt_date"] - df["time_early_dt"]).dt.days
        df["days_to_bankrupt_late"] = (df["bankrupt_date"] - df["time_late_dt"]).dt.days
    else:
        df["days_to_bankrupt_early"] = np.nan
        df["days_to_bankrupt_late"] = np.nan

    # 确保 early/late 关键列存在
    for col in ["speaker_early", "speaker_late"]:
        if col not in df.columns:
            df[col] = "UNKNOWN"
        df[col] = df[col].fillna("UNKNOWN").astype(str).str.strip()

    for col in ["dominant_cause_early", "dominant_cause_late"]:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].apply(normalize_risk_bucket)

    if "early_n_sent" not in df.columns:
        df["early_n_sent"] = 0.0
    if "late_n_sent" not in df.columns:
        df["late_n_sent"] = 0.0

    df["early_n_sent"] = pd.to_numeric(df["early_n_sent"], errors="coerce").fillna(0.0)
    df["late_n_sent"] = pd.to_numeric(df["late_n_sent"], errors="coerce").fillna(0.0)

    unit = unit_logit_df.copy()

    # early unit logit
    df_early = df.merge(
        unit.rename(columns={
            "role": "speaker_early",
            "risk_bucket": "dominant_cause_early",
            "unit_logit": "unit_logit_early",
        }),
        left_on=["speaker_early", "dominant_cause_early", "z_label_true"],
        right_on=["speaker_early", "dominant_cause_early", "z_label"],
        how="left",
    )

    df_early["unit_logit_early"] = df_early.get("unit_logit_early", np.nan)
    df_early["unit_logit_early"] = df_early["unit_logit_early"].fillna(0.0)

    if "z_label" in df_early.columns:
        df_early.drop(columns=["z_label"], inplace=True)

    df_early["C_logit_early"] = df_early["unit_logit_early"] * df_early["early_n_sent"]

    # late unit logit
    df_late = df_early.merge(
        unit.rename(columns={
            "role": "speaker_late",
            "risk_bucket": "dominant_cause_late",
            "unit_logit": "unit_logit_late",
        }),
        left_on=["speaker_late", "dominant_cause_late", "z_label_true"],
        right_on=["speaker_late", "dominant_cause_late", "z_label"],
        how="left",
    )

    df_late["unit_logit_late"] = df_late.get("unit_logit_late", np.nan)
    df_late["unit_logit_late"] = df_late["unit_logit_late"].fillna(0.0)

    if "z_label" in df_late.columns:
        df_late.drop(columns=["z_label"], inplace=True)

    df_late["C_logit_late"] = df_late["unit_logit_late"] * df_late["late_n_sent"]

    # 公司概率用 company_norm 合并
    probs = company_probs.rename(columns={"pmz": "p_mz"}).copy()

    df_prob = df_late.merge(
        probs,
        left_on=["company_norm", "z_label_true"],
        right_on=["company_norm", "z_label"],
        how="left",
    )

    df_prob["p_mz"] = df_prob.get("p_mz", np.nan)
    missing_p = df_prob["p_mz"].isna().sum()
    if missing_p > 0:
        logger.warning(
            "[prob] %d rows missing p_mz after company_norm merge. "
            "These rows will get p_mz=0 and may yield zero importance.",
            missing_p,
        )
    df_prob["p_mz"] = df_prob["p_mz"].fillna(0.0)

    if "z_label" in df_prob.columns:
        df_prob.drop(columns=["z_label"], inplace=True)

    # importance
    df_prob["importance_early"] = df_prob["p_mz"] * (1.0 - df_prob["p_mz"]) * df_prob["C_logit_early"]
    df_prob["importance_late"] = df_prob["p_mz"] * (1.0 - df_prob["p_mz"]) * df_prob["C_logit_late"]
    df_prob["importance_delta"] = df_prob["importance_late"] - df_prob["importance_early"]

    detailed_df = df_prob.copy()

    # summary
    group_cols = ["company", "term", "source_type"]
    agg_rows = []

    for (m, term, z_str), g in detailed_df.groupby(group_cols):
        imp_early = g["importance_early"].mean()
        imp_late = g["importance_late"].mean()
        imp_delta = g["importance_delta"].mean()

        cause_early = mode_or_unknown(g["dominant_cause_early"])
        cause_late = mode_or_unknown(g["dominant_cause_late"])
        speaker_early = mode_or_unknown(g["speaker_early"])
        speaker_late = mode_or_unknown(g["speaker_late"])

        ts_mean = g["term_score"].mean() if "term_score" in g.columns else np.nan

        bd = g["bankrupt_date"].dropna() if "bankrupt_date" in g.columns else pd.Series([], dtype="datetime64[ns]")
        bankrupt_date_val = bd.iloc[0] if not bd.empty else pd.NaT

        # 新增：time_early/time_late 和 early_n_sent/late_n_sent 聚合
        if "time_early_dt" in g.columns:
            time_early_val = g["time_early_dt"].min()
        else:
            time_early_val = pd.NaT

        if "time_late_dt" in g.columns:
            time_late_val = g["time_late_dt"].max()
        else:
            time_late_val = pd.NaT

        early_n_sent_total = float(g["early_n_sent"].sum())
        late_n_sent_total = float(g["late_n_sent"].sum())

        agg_rows.append(
            {
                "company": m,
                "term": term,
                "risk_type_z": z_str,
                "imp_early": float(imp_early),
                "imp_late": float(imp_late),
                "imp_delta": float(imp_delta),
                "term_score_mean": float(ts_mean) if not np.isnan(ts_mean) else np.nan,
                "mode_speaker_early": speaker_early,
                "mode_speaker_late": speaker_late,
                "mode_dominant_cause_early": cause_early,
                "mode_dominant_cause_late": cause_late,
                "n_rows": int(len(g)),
                "bankrupt_date": bankrupt_date_val,
                "time_early": time_early_val,
                "time_late": time_late_val,
                "early_n_sent": early_n_sent_total,
                "late_n_sent": late_n_sent_total,
            }
        )

    summary_df = pd.DataFrame(agg_rows)

    logger.info(
        "Constructed detailed term importance table (rows=%d) and summary (rows=%d).",
        len(detailed_df),
        len(summary_df),
    )

    return detailed_df, summary_df


# -----------------------------
# 主入口
# -----------------------------

def main():
    logger.info("===== Step 5.3 Term importance START =====")

    # 1) 读入 5.2 输出
    df_recurring = load_and_filter_recurring_terms(RECURRING_TERMS_PATH)
    if df_recurring.empty:
        logger.error("No data available after filtering recurring terms.")
        return

    # 2) 读入 bankrupt 并多级匹配
    bank_df = load_bankrupt_table(BANKRUPT_CSV)
    df_recurring = attach_bankrupt_date(df_recurring, bank_df)

    # 3) 读入 RQ1 surrogate 组件
    f_pvt, beta_map, gamma_map, company_probs = load_predict_components()
    if f_pvt.empty or not beta_map or company_probs.empty:
        logger.error("Failed to load RQ1 components.")
        return

    # 4) 构造 unit_logit
    unit_logit_df = build_unit_logit_table(f_pvt, beta_map, gamma_map)
    if unit_logit_df.empty:
        logger.error("unit_logit_df is empty.")
        return

    # 5) 计算重要性
    detailed_df, summary_df = compute_term_importance(
        df_recurring,
        unit_logit_df,
        company_probs,
    )

    if detailed_df.empty or summary_df.empty:
        logger.warning("Empty outputs after compute_term_importance.")
        return

    # 6) 保存 detailed（原样）
    logger.info("Saving detailed term importance to %s", DETAILED_OUT)
    detailed_df.to_csv(DETAILED_OUT, index=False)

    # 7) res_table（不删）= 原始 summary
    res_table = summary_df.copy()
    logger.info("Saving term importance summary (raw) to %s", SUMMARY_OUT)
    res_table.to_csv(SUMMARY_OUT, index=False)

    # 8) res_table_clean
    res_table_clean = res_table.copy()

    # (1) 删掉 bankrupt_date 匹配不上的行
    if "bankrupt_date" in res_table_clean.columns:
        before = len(res_table_clean)
        res_table_clean = res_table_clean[res_table_clean["bankrupt_date"].notna()].copy()
        logger.info(
            "[clean] Drop rows with missing bankrupt_date: %d -> %d",
            before, len(res_table_clean)
        )
    else:
        logger.warning("[clean] 'bankrupt_date' not found in summary, skip rule (1).")

    # (2) 删掉 imp_early/imp_late/imp_delta 全为 0 的行
    for c in ["imp_early", "imp_late", "imp_delta"]:
        if c not in res_table_clean.columns:
            res_table_clean[c] = 0.0

    zero_mask = (
        np.isclose(res_table_clean["imp_early"].fillna(0.0), 0.0) &
        np.isclose(res_table_clean["imp_late"].fillna(0.0), 0.0) &
        np.isclose(res_table_clean["imp_delta"].fillna(0.0), 0.0)
    )
    before = len(res_table_clean)
    res_table_clean = res_table_clean.loc[~zero_mask].copy()
    logger.info(
        "[clean] Drop rows with imp_early/imp_late/imp_delta all zero: %d -> %d",
        before, len(res_table_clean)
    )

    # 9) 保存 clean 表
    logger.info("Saving term importance summary (clean) to %s", SUMMARY_CLEAN_OUT)
    res_table_clean.to_csv(SUMMARY_CLEAN_OUT, index=False)

    logger.info("===== Step 5.3 Term importance DONE =====")


if __name__ == "__main__":
    main()
