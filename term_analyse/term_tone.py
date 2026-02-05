# term_tone.py
# -*- coding: utf-8 -*-
"""
Step 5.2 —— Term-level 跨期“内部风险担忧语气”轨迹 + 显著变化筛选（带 speaker / j / z）

输入
1) term set（5.1 输出）
    /home/ubuntu/project/term_analyse/output/terms_selected_from_raw.json

2) 按句子打好分、带 RQ1 风险后验、speaker、source_type 的主表
    /home/ubuntu/project/identification/output/sentences_scored_llm_ctx.csv

输出（都在 term_analyse/output/step_5_2_term_tone_shift/ 下）
- term_sentence_occurrences.csv
    每条命中 term 的句子一行，包含 company / file_date / speaker / source_type / 风险后验等
- term_window_tone.csv
    公司–term–speaker–时间窗口(early/mid/late) 的平均语气
- term_cross_period_shifts_by_speaker.csv
    公司–term–speaker 级别的 early vs late 语气差分 + 主导成因 early/late
- recurring_terms_with_significant_tone_shift.csv
    公司–term 级别综合分数
        D_max, D_med, term_score, early/late 覆盖度,
        主导 speaker（整体）、前期主导 speaker、后期主导 speaker，
        主导成因 early/late, z（破产类型），
        time_early  early 阶段 term 出现高峰日期（YYYY-MM-DD）
        time_late   late  阶段 term 出现高峰日期（YYYY-MM-DD）

额外要求
- 不生成新的 stats 文件
- 将 Step 5.2 的 term 统计直接追加到 build_term 的 term_build_stats.json 里（就地更新）

当前版本的门槛（较原版略放宽）
- company–term–speaker–stage 窗口内句子数阈值  min_sent_per_window = 5
- company–term 层面的 early_n_sent / late_n_sent 覆盖度阈值  均为 5
- 显著性分位数  term_score ≥ 80% 分位视为显著  significant_quantile = 0.8
"""

import os
import re
import json
import math
import logging
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Tuple, Any

import pandas as pd

# -----------------------------
# 路径配置
# -----------------------------

PROJ_ROOT = r"D:\UOW\risk idenfitication\code\risk-identification"

TERM_SET_PATH = os.path.join(
    PROJ_ROOT, "term_analyse", "output", "terms_selected_from_raw.json"
)

SENT_CTX_PATH = os.path.join(
    PROJ_ROOT, "identification", "output", "sentences_scored_llm_ctx.csv"
)

OUT_ROOT = os.path.join(
    PROJ_ROOT, "term_analyse", "output", "step_5_2_term_tone_shift"
)
os.makedirs(OUT_ROOT, exist_ok=True)

TERM_SENTENCE_CSV = os.path.join(OUT_ROOT, "term_sentence_occurrences.csv")
TERM_WINDOW_TONE_CSV = os.path.join(OUT_ROOT, "term_window_tone.csv")
TERM_SHIFT_CSV = os.path.join(
    OUT_ROOT, "term_cross_period_shifts_by_speaker.csv"
)
TERM_FINAL_SCORE_CSV = os.path.join(
    OUT_ROOT, "recurring_terms_with_significant_tone_shift.csv"
)

# ★ 关键 直接更新 build_term 的统计文件
TERM_BUILD_STATS_JSON = os.path.join(
    PROJ_ROOT, "term_analyse", "output", "term_build_stats.json"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("step_5_2_term_tone_shift")

# -----------------------------
# Tone 维度配置
# -----------------------------
RISK_CAUSE_COLS = [
    "prob_risk_uncertainty",
    "prob_risk_legal",
    "prob_risk_constraint",
    "prob_risk_external",
]
RISK_CAUSE_LABELS = [
    "uncertainty",
    "legal",
    "constraint",
    "external",
]

TONE_COLS = [
    *RISK_CAUSE_COLS,
    "llm_pred_prob",
    "risk_lexical_intensity",
    "sentiment_score",
    "uncertainty_score",
    "hedging_score",
    "commitment_score",
]

# -----------------------------
# 词典
# -----------------------------
RISK_WORDS = {
    "liquidity", "cash", "covenant", "default", "insolvency", "bankruptcy",
    "going concern", "refinance", "refinancing", "downgrade", "impairment",
    "write-down", "writeoff", "write-off", "distress", "restructuring",
    "covenants", "breach", "breaches", "loss", "losses",
    "capital", "leverage", "debt", "obligation", "exposure", "volatility",
}

POS_WORDS = {
    "strong", "improve", "improved", "improving", "confident", "optimistic",
    "opportunity", "resilient", "solid", "robust", "comfortable",
}

NEG_WORDS = {
    "weak", "deteriorate", "deteriorating", "worse", "worsening", "concern",
    "concerns", "risk", "risks", "pressure", "pressures", "challenge",
    "challenges", "headwind", "headwinds", "uncertain", "uncertainty",
    "doubt", "doubts", "worry", "worried", "anxious",
}

UNCERTAINTY_WORDS = {
    "maybe", "might", "possibly", "uncertain", "uncertainty", "unknown",
    "depends", "depending", "roughly", "approximately", "around", "about",
}

HEDGING_WORDS = {
    "subject to", "as long as", "as far as", "unless", "provided that",
    "contingent on", "depending on",
}

COMMITMENT_WORDS = {
    "will", "committed", "commit", "guarantee", "guaranteed",
    "definitely", "certainly", "assure", "assured",
}

# -----------------------------
# 数据结构
# -----------------------------
@dataclass
class TermSentenceRecord:
    company: str
    file_date: datetime
    year: int
    quarter_label: str
    company_time_index: int
    company_stage: str
    speaker: str
    source_type: str
    sentence_id: int
    sentence_text: str
    term: str

    prob_risk_uncertainty: float
    prob_risk_legal: float
    prob_risk_constraint: float
    prob_risk_external: float
    llm_pred_prob: float

    risk_lexical_intensity: float
    sentiment_score: float
    uncertainty_score: float
    hedging_score: float
    commitment_score: float

# -----------------------------
# 工具函数
# -----------------------------
def load_term_set(term_path: str) -> List[str]:
    logger.info(f"Loading term set from {term_path}")
    with open(term_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    terms: List[str] = []
    if isinstance(data, list):
        terms = [str(t).strip() for t in data if str(t).strip()]
    elif isinstance(data, dict):
        if "terms" in data and isinstance(data["terms"], list):
            terms = [str(t).strip() for t in data["terms"] if str(t).strip()]
        elif "selected_terms" in data and isinstance(data["selected_terms"], list):
            terms = [str(t).strip() for t in data["selected_terms"] if str(t).strip()]
        else:
            terms = [str(k).strip() for k in data.keys() if str(k).strip()]
    else:
        raise ValueError("Unsupported term set JSON structure")

    terms = sorted(set(terms))
    logger.info(f"Loaded {len(terms)} unique terms from term set")
    return terms


def build_term_index(terms: List[str]):
    """
    将 term 拆成 token n-gram 索引，减少一句乘全部 term 正则的开销
    """
    ngram_index: Dict[int, Dict[str, List[str]]] = {}
    complex_patterns: Dict[str, re.Pattern] = {}

    alpha_space_pattern = re.compile(r"^[A-Za-z\s]+$")

    for term in terms:
        t_raw = term.strip()
        if not t_raw:
            continue
        t_lower = t_raw.lower()

        if alpha_space_pattern.match(t_lower):
            tokens = re.findall(r"[a-zA-Z]+", t_lower)
            if not tokens:
                continue
            n = len(tokens)
            key = " ".join(tokens)
            if n not in ngram_index:
                ngram_index[n] = {}
            ngram_index[n].setdefault(key, []).append(t_raw)
        else:
            escaped = re.escape(t_lower)
            pattern = re.compile(r"\b" + escaped + r"\b", flags=re.IGNORECASE)
            complex_patterns[t_raw] = pattern

    total_ngram_terms = sum(len(v) for d in ngram_index.values() for v in d.values())
    logger.info(
        f"Built n-gram index for {total_ngram_terms} terms; "
        f"{len(complex_patterns)} complex terms kept as regex"
    )
    return ngram_index, complex_patterns


def find_terms_in_sentence(
    sentence_lower: str,
    tokens: List[str],
    ngram_index: Dict[int, Dict[str, List[str]]],
    complex_patterns: Dict[str, re.Pattern],
) -> List[str]:
    found = set()
    token_len = len(tokens)

    for n, ngram_dict in ngram_index.items():
        if token_len < n:
            continue
        for i in range(token_len - n + 1):
            ngram = " ".join(tokens[i: i + n])
            if ngram in ngram_dict:
                for original_term in ngram_dict[ngram]:
                    found.add(original_term)

    for term, pattern in complex_patterns.items():
        if pattern.search(sentence_lower):
            found.add(term)

    return list(found)


_DOC_PARSE_CACHE: Dict[str, Tuple[str, datetime, str]] = {}


def parse_company_date_quarter(doc_id: str) -> Tuple[str, datetime, str]:
    """
    预期格式
        Company Name_YYYY-MM-DD_Qx YYYY
    """
    if doc_id in _DOC_PARSE_CACHE:
        return _DOC_PARSE_CACHE[doc_id]

    base = doc_id
    m = re.match(r"^(.*)_(\d{4}-\d{2}-\d{2})_Q(\d)\s*(\d{4})$", base)
    if not m:
        logger.warning(f"doc_id format unexpected, using fallback parsing: {doc_id}")
        parts = base.split("_")
        date_str = None
        for p in parts:
            if re.match(r"\d{4}-\d{2}-\d{2}", p):
                date_str = p
                break
        if date_str is None:
            raise ValueError(f"Cannot parse date from doc_id {doc_id}")
        date = datetime.strptime(date_str, "%Y-%m-%d")
        company = base.replace(f"_{date_str}", "")
        quarter_label = f"Q? {date.year}"
        _DOC_PARSE_CACHE[doc_id] = (company, date, quarter_label)
        return company, date, quarter_label

    company = m.group(1)
    date_str = m.group(2)
    q_num = m.group(3)
    year_str = m.group(4)

    date = datetime.strptime(date_str, "%Y-%m-%d")
    quarter_label = f"Q{q_num} {year_str}"

    _DOC_PARSE_CACHE[doc_id] = (company, date, quarter_label)
    return company, date, quarter_label


def parse_company_date_quarter_from_source(path_or_id: str) -> Tuple[str, datetime, str]:
    base = os.path.basename(str(path_or_id))
    base = re.sub(r"\.sentences\.csv$", "", base)
    base = re.sub(r"\.csv$", "", base)
    base = re.sub(r"\.txt$", "", base)
    return parse_company_date_quarter(base)


def compute_lexical_tone_features(sentence: str) -> Dict[str, float]:
    s_lower = sentence.lower()
    tokens = re.findall(r"[a-zA-Z]+", s_lower)
    n_tokens = len(tokens)
    if n_tokens == 0:
        return dict(
            risk_lexical_intensity=0.0,
            sentiment_score=0.0,
            uncertainty_score=0.0,
            hedging_score=0.0,
            commitment_score=0.0,
        )

    token_set = set(tokens)

    risk_count = 0
    for w in RISK_WORDS:
        if " " in w:
            if w in s_lower:
                risk_count += 1
        else:
            if w in token_set:
                risk_count += 1

    pos_count = sum(1 for w in tokens if w in POS_WORDS)
    neg_count = sum(1 for w in tokens if w in NEG_WORDS)
    sentiment = (pos_count - neg_count) / (n_tokens + 1.0)

    unc_count = sum(1 for w in tokens if w in UNCERTAINTY_WORDS)

    hedging_hit = 0
    for phrase in HEDGING_WORDS:
        if phrase in s_lower:
            hedging_hit = 1
            break

    commit_count = sum(1 for w in tokens if w in COMMITMENT_WORDS)

    return dict(
        risk_lexical_intensity=risk_count / (n_tokens + 1.0),
        sentiment_score=sentiment,
        uncertainty_score=unc_count / (n_tokens + 1.0),
        hedging_score=float(hedging_hit),
        commitment_score=commit_count / (n_tokens + 1.0),
    )


def safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
        if math.isnan(v):
            return default
        return v
    except Exception:
        return default


def _term_ngram_len(term: str) -> int:
    """
    统计 term 的字母 token 数
    """
    if term is None:
        return 0
    tokens = re.findall(r"[A-Za-z]+", str(term).strip())
    return len(tokens)


def update_json_inplace(path: str, patch: Dict[str, Any]) -> None:
    """
    就地更新 JSON 文件
    - 文件必须已存在
    - 保留原字段，只新增或覆盖 patch 中的字段
    """
    if not os.path.isfile(path):
        logger.error(f"Target JSON not found, will not create a new one: {path}")
        return

    data: Dict[str, Any] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        if isinstance(loaded, dict):
            data = loaded
        else:
            logger.warning(f"JSON root is not an object, overwrite with object. path={path}")
            data = {}
    except Exception as e:
        logger.error(f"Failed to read JSON, skip updating. path={path}, err={e}")
        return

    data.update(patch)

    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        logger.info(f"Updated JSON in place: {path}")
    except Exception as e:
        logger.error(f"Failed to write JSON. path={path}, err={e}")


# -----------------------------
# 1 扫描 term 命中
# -----------------------------
def scan_sentences_from_llm_ctx(chunk_size: int = 50_000) -> pd.DataFrame:
    if not os.path.isfile(SENT_CTX_PATH):
        logger.error(f"Sentence ctx CSV not found: {SENT_CTX_PATH}")
        return pd.DataFrame()

    terms = load_term_set(TERM_SET_PATH)
    ngram_index, complex_patterns = build_term_index(terms)

    logger.info(
        f"Scanning sentences from {SENT_CTX_PATH} "
        f"with chunk_size={chunk_size}"
    )

    records: List[TermSentenceRecord] = []

    total_rows = 0
    total_term_hits = 0
    total_chunks = 0

    required_cols = [
        "doc_id", "text", "sent_idx",
        "speaker_role",
        "source_type",
        "prob_risk_uncertainty", "prob_risk_legal",
        "prob_risk_constraint", "prob_risk_external",
        "llm_pred_prob",
    ]

    for chunk in pd.read_csv(SENT_CTX_PATH, chunksize=chunk_size):
        total_chunks += 1
        logger.info(
            f"Processing chunk {total_chunks}, rows in this chunk = {len(chunk)} "
            f"(total_rows so far = {total_rows})"
        )

        for col in required_cols:
            if col not in chunk.columns:
                logger.error(f"Required column '{col}' not found in chunk. Aborting.")
                return pd.DataFrame()

        has_source_path = "source_csv_path" in chunk.columns

        for _, row in chunk.iterrows():
            total_rows += 1

            doc_key = None
            if has_source_path:
                src = row.get("source_csv_path", None)
                if src is not None and str(src).strip():
                    doc_key = str(src)
            if doc_key is None:
                doc_key = str(row["doc_id"])

            try:
                company, file_date, quarter_label = parse_company_date_quarter_from_source(doc_key)
            except Exception as e:
                logger.error(f"Failed to parse doc key={doc_key}: {e}")
                continue

            text = str(row["text"]) if not pd.isna(row["text"]) else ""
            if not text.strip():
                continue

            speaker = str(row.get("speaker_role", "UNKNOWN")) or "UNKNOWN"
            source_type = str(row.get("source_type", "")).strip()
            sent_id = int(row.get("sent_idx", -1))

            prob_risk_uncertainty = safe_float(row.get("prob_risk_uncertainty", 0.0), 0.0)
            prob_risk_legal = safe_float(row.get("prob_risk_legal", 0.0), 0.0)
            prob_risk_constraint = safe_float(row.get("prob_risk_constraint", 0.0), 0.0)
            prob_risk_external = safe_float(row.get("prob_risk_external", 0.0), 0.0)
            llm_pred_prob = safe_float(row.get("llm_pred_prob", 0.0), 0.0)

            lex_tone = compute_lexical_tone_features(text)

            s_lower = text.lower()
            tokens = re.findall(r"[a-zA-Z]+", s_lower)

            matched_terms = find_terms_in_sentence(
                s_lower, tokens, ngram_index, complex_patterns
            )
            if not matched_terms:
                continue

            for term in matched_terms:
                total_term_hits += 1
                rec = TermSentenceRecord(
                    company=company,
                    file_date=file_date,
                    year=file_date.year,
                    quarter_label=quarter_label,
                    company_time_index=-1,
                    company_stage="UNKNOWN",
                    speaker=speaker,
                    source_type=source_type,
                    sentence_id=sent_id,
                    sentence_text=text,
                    term=term,
                    prob_risk_uncertainty=prob_risk_uncertainty,
                    prob_risk_legal=prob_risk_legal,
                    prob_risk_constraint=prob_risk_constraint,
                    prob_risk_external=prob_risk_external,
                    llm_pred_prob=llm_pred_prob,
                    risk_lexical_intensity=lex_tone["risk_lexical_intensity"],
                    sentiment_score=lex_tone["sentiment_score"],
                    uncertainty_score=lex_tone["uncertainty_score"],
                    hedging_score=lex_tone["hedging_score"],
                    commitment_score=lex_tone["commitment_score"],
                )
                records.append(rec)

            if total_rows % 10_000 == 0:
                logger.info(
                    f"Scanned {total_rows} sentence rows so far; "
                    f"current term hits = {total_term_hits}"
                )

    logger.info(
        f"Finished scanning sentences. Total rows = {total_rows}, "
        f"total term hits = {total_term_hits}"
    )

    if not records:
        logger.warning("No term occurrences found from sentences_scored_llm_ctx.")
        return pd.DataFrame()

    df = pd.DataFrame([asdict(r) for r in records])
    return df


# -----------------------------
# 2 公司内部时间序列 early mid late
# -----------------------------
def assign_company_time_indices(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    logger.info("Assigning company_time_index and company_stage (early/mid/late)")
    df = df.copy()
    df.sort_values(["company", "file_date"], inplace=True)

    time_index_map = {}
    stage_map = {}

    for company, group in df.groupby("company"):
        unique_dates = sorted(group["file_date"].unique())
        n_periods = len(unique_dates)
        if n_periods == 0:
            continue

        early_end = max(0, n_periods // 3 - 1)
        mid_end = max(early_end + 1, 2 * n_periods // 3 - 1)

        logger.info(
            f"Company {company}: {n_periods} periods, "
            f"early=[0,{early_end}], mid=[{early_end+1},{mid_end}], "
            f"late=[{mid_end+1},{n_periods-1}]"
        )

        for idx, d in enumerate(unique_dates):
            time_index_map[(company, d)] = idx
            if idx <= early_end:
                stage = "early"
            elif idx <= mid_end:
                stage = "mid"
            else:
                stage = "late"
            stage_map[(company, d)] = stage

    df["company_time_index"] = df.apply(
        lambda row: time_index_map.get((row["company"], row["file_date"]), -1),
        axis=1,
    )
    df["company_stage"] = df.apply(
        lambda row: stage_map.get((row["company"], row["file_date"]), "UNKNOWN"),
        axis=1,
    )

    unknown_count = (df["company_time_index"] < 0).sum()
    if unknown_count > 0:
        logger.warning(f"{unknown_count} rows have unknown company_time_index")

    return df


# -----------------------------
# 3 聚合 company term speaker window
# -----------------------------
def aggregate_term_tone_by_window(
    df: pd.DataFrame,
    min_sent_per_window: int = 5
) -> pd.DataFrame:
    if df.empty:
        return df

    logger.info("Aggregating tone features by company-term-speaker-window")

    group_cols = ["company", "term", "speaker", "company_stage"]
    agg_funcs = {col: "mean" for col in TONE_COLS}
    agg_funcs["sentence_id"] = "count"

    grouped = df.groupby(group_cols).agg(agg_funcs).reset_index()
    grouped.rename(columns={"sentence_id": "n_sentences"}, inplace=True)

    before_filter = len(grouped)
    grouped = grouped[grouped["n_sentences"] >= min_sent_per_window].copy()
    after_filter = len(grouped)

    logger.info(
        f"Window-level term tone: {before_filter} groups before filter, "
        f"{after_filter} groups with n_sentences >= {min_sent_per_window}"
    )

    return grouped


# -----------------------------
# 4 跨期差分 early vs late
# -----------------------------
def compute_cross_period_shifts(window_df: pd.DataFrame) -> pd.DataFrame:
    if window_df.empty:
        return pd.DataFrame()

    logger.info("Computing cross-period shifts for each company-term-speaker")

    wdf = window_df[window_df["company_stage"].isin(["early", "late"])].copy()
    if wdf.empty:
        logger.warning("No early/late window data found, cannot compute shifts")
        return pd.DataFrame()

    pivoted = wdf.pivot_table(
        index=["company", "term", "speaker"],
        columns="company_stage",
        values=TONE_COLS,
    )
    pivoted.columns = [f"{c[0]}_{c[1]}" for c in pivoted.columns]
    pivoted = pivoted.reset_index()

    early_cols = [f"{c}_early" for c in TONE_COLS]
    late_cols = [f"{c}_late" for c in TONE_COLS]
    mask_has_early = pivoted[early_cols].notna().all(axis=1)
    mask_has_late = pivoted[late_cols].notna().all(axis=1)
    valid = pivoted[mask_has_early & mask_has_late].copy()

    logger.info(
        f"Total company-term-speaker combos: {len(pivoted)}, "
        f"with both early & late tone: {len(valid)}"
    )
    if valid.empty:
        return pd.DataFrame()

    std_map: Dict[str, float] = {}
    for col in TONE_COLS:
        all_vals = pd.concat([valid[f"{col}_early"], valid[f"{col}_late"]], axis=0)
        std = all_vals.std()
        if std <= 1e-6:
            std = 1.0
        std_map[col] = std

    def compute_D(row) -> float:
        sq_sum = 0.0
        for col in TONE_COLS:
            early_val = row[f"{col}_early"]
            late_val = row[f"{col}_late"]
            delta = (late_val - early_val) / std_map[col]
            sq_sum += delta * delta
        return math.sqrt(sq_sum)

    valid["tone_shift_D"] = valid.apply(compute_D, axis=1)

    def argmax_cause(row, suffix: str) -> str:
        probs = [row[f"{c}_{suffix}"] for c in RISK_CAUSE_COLS]
        idx = max(range(len(probs)), key=lambda k: probs[k])
        return RISK_CAUSE_LABELS[idx]

    valid["dominant_cause_early"] = valid.apply(
        lambda r: argmax_cause(r, "early"), axis=1
    )
    valid["dominant_cause_late"] = valid.apply(
        lambda r: argmax_cause(r, "late"), axis=1
    )

    logger.info(
        "Computed tone_shift_D and dominant_cause_early/late for all valid company-term-speaker combos. "
        f"mean D={valid['tone_shift_D'].mean():.4f}, "
        f"median D={valid['tone_shift_D'].median():.4f}, "
        f"95pct D={valid['tone_shift_D'].quantile(0.95):.4f}"
    )

    return valid


# -----------------------------
# 5A 计算 peak date
# -----------------------------
def compute_peak_date_for_terms(df_terms: pd.DataFrame) -> pd.DataFrame:
    if df_terms.empty or "company_stage" not in df_terms.columns:
        return pd.DataFrame(columns=["company", "term", "time_early", "time_late"])

    df = df_terms[df_terms["company_stage"].isin(["early", "late"])].copy()
    if df.empty:
        return pd.DataFrame(columns=["company", "term", "time_early", "time_late"])

    df["file_date"] = pd.to_datetime(df["file_date"])

    cnt = df.groupby(
        ["company", "term", "company_stage", "file_date"]
    ).size().reset_index(name="n_sent")

    if cnt.empty:
        return pd.DataFrame(columns=["company", "term", "time_early", "time_late"])

    cnt = cnt.sort_values(
        ["company", "term", "company_stage", "n_sent", "file_date"],
        ascending=[True, True, True, False, True],
    )
    peak = cnt.drop_duplicates(["company", "term", "company_stage"])

    pivot = peak.pivot(
        index=["company", "term"],
        columns="company_stage",
        values="file_date",
    ).reset_index()

    if "early" in pivot.columns:
        pivot = pivot.rename(columns={"early": "time_early"})
        pivot["time_early"] = pivot["time_early"].dt.strftime("%Y-%m-%d")
    else:
        pivot["time_early"] = ""

    if "late" in pivot.columns:
        pivot = pivot.rename(columns={"late": "time_late"})
        pivot["time_late"] = pivot["time_late"].dt.strftime("%Y-%m-%d")
    else:
        pivot["time_late"] = ""

    if "time_early" not in pivot.columns:
        pivot["time_early"] = ""
    if "time_late" not in pivot.columns:
        pivot["time_late"] = ""

    return pivot[["company", "term", "time_early", "time_late"]]


# -----------------------------
# 5B term score
# -----------------------------
def aggregate_term_scores(
    shift_df: pd.DataFrame,
    window_df: pd.DataFrame,
    df_terms: pd.DataFrame,
    significant_quantile: float = 0.8
) -> pd.DataFrame:
    if shift_df.empty:
        logger.warning("No shift data provided to aggregate_term_scores")
        return pd.DataFrame()

    logger.info("Aggregating term-level scores per company-term (with speaker info)")

    agg = shift_df.groupby(["company", "term"]).agg(
        D_max=("tone_shift_D", "max"),
        D_med=("tone_shift_D", "median"),
        speakers_count=("speaker", "nunique"),
    ).reset_index()

    idx_max = shift_df.groupby(["company", "term"])["tone_shift_D"].idxmax()
    main_df = shift_df.loc[idx_max, [
        "company",
        "term",
        "speaker",
        "dominant_cause_early",
        "dominant_cause_late",
    ]].copy()

    main_df = main_df.rename(columns={"speaker": "main_speaker_role"})
    main_df["main_speaker_role_early"] = main_df["main_speaker_role"]
    main_df["main_speaker_role_late"] = main_df["main_speaker_role"]

    agg = agg.merge(main_df, on=["company", "term"], how="left")

    for col in ["D_max", "D_med"]:
        mean_val = agg[col].mean()
        std_val = agg[col].std()
        if std_val <= 1e-6:
            std_val = 1.0
        agg[f"{col}_z"] = (agg[col] - mean_val) / std_val

    agg["term_score"] = 0.7 * agg["D_max_z"] + 0.3 * agg["D_med_z"]

    logger.info("Filtering company-term combinations based on early/late window activity")

    wdf = window_df[window_df["company_stage"].isin(["early", "late"])].copy()
    if wdf.empty:
        logger.warning("No early/late window_df, skipping coverage-based filter")
        agg["early_n_sent"] = 0
        agg["late_n_sent"] = 0
    else:
        w_grp = wdf.groupby(["company", "term", "company_stage"])["n_sentences"].sum().reset_index()
        w_pivot = w_grp.pivot_table(
            index=["company", "term"],
            columns="company_stage",
            values="n_sentences",
        ).reset_index()

        if ("early" in w_pivot.columns) or ("late" in w_pivot.columns):
            w_pivot.rename(
                columns={"early": "early_n_sent", "late": "late_n_sent"},
                inplace=True,
            )
        else:
            rename_map = {}
            for c in w_pivot.columns:
                if isinstance(c, tuple) and c[1] in ("early", "late"):
                    rename_map[c] = f"{c[1]}_n_sent"
            if rename_map:
                w_pivot.rename(columns=rename_map, inplace=True)

        agg = agg.merge(w_pivot, on=["company", "term"], how="left")

        if "early_n_sent" not in agg.columns:
            agg["early_n_sent"] = 0
        if "late_n_sent" not in agg.columns:
            agg["late_n_sent"] = 0

        MIN_EARLY = 5
        MIN_LATE = 5
        coverage_mask = (agg["early_n_sent"] >= MIN_EARLY) & (agg["late_n_sent"] >= MIN_LATE)
        before_cov = len(agg)
        agg = agg[coverage_mask].copy()
        after_cov = len(agg)
        logger.info(
            f"Coverage filter: {before_cov} company-term before, "
            f"{after_cov} with early_n_sent >= {MIN_EARLY} "
            f"and late_n_sent >= {MIN_LATE}"
        )

    if agg.empty:
        logger.warning("No company-term left after coverage filter.")
        return agg

    try:
        peak_date_df = compute_peak_date_for_terms(df_terms)
        if not peak_date_df.empty:
            agg = agg.merge(peak_date_df, on=["company", "term"], how="left")
        else:
            agg["time_early"] = ""
            agg["time_late"] = ""
    except Exception as e:
        logger.error(f"Failed to compute time_early/time_late: {e}")
        agg["time_early"] = ""
        agg["time_late"] = ""

    if "time_early" not in agg.columns:
        agg["time_early"] = ""
    if "time_late" not in agg.columns:
        agg["time_late"] = ""

    if "true_z" in df_terms.columns:
        company_info = df_terms[["company", "source_type", "true_z"]].drop_duplicates()
        company_info = company_info.rename(columns={"true_z": "z"})
    else:
        company_info = df_terms[["company", "source_type"]].drop_duplicates()
        company_info["z"] = company_info["source_type"]

    agg = agg.merge(company_info, on="company", how="left")

    score_threshold = agg["term_score"].quantile(significant_quantile)
    agg["is_significant"] = agg["term_score"] >= score_threshold

    logger.info(
        f"Term_score threshold at quantile {significant_quantile:.2f}: "
        f"{score_threshold:.4f}. "
        f"{agg['is_significant'].sum()} company-term combos flagged as significant."
    )

    return agg


def _count_unique_unigram_bigram(terms: List[str]) -> Dict[str, int]:
    uniq = sorted(set([t for t in terms if t is not None and str(t).strip()]))
    uni = 0
    bi = 0
    for t in uniq:
        n = _term_ngram_len(t)
        if n == 1:
            uni += 1
        elif n == 2:
            bi += 1
    return {
        "unique_terms_total": int(len(uniq)),
        "unique_unigram_terms": int(uni),
        "unique_bigram_terms": int(bi),
    }


# -----------------------------
# 主入口
# -----------------------------
def main():
    logger.info("===== Step 5.2: Term-level cross-period tone shift analysis START =====")

    df_terms = scan_sentences_from_llm_ctx(chunk_size=50_000)
    if df_terms.empty:
        logger.error("No term occurrences found. Aborting 5.2 pipeline.")
        return

    df_terms = assign_company_time_indices(df_terms)

    logger.info(f"Saving term sentence occurrences to {TERM_SENTENCE_CSV}")
    df_terms.to_csv(TERM_SENTENCE_CSV, index=False)

    window_df = aggregate_term_tone_by_window(df_terms, min_sent_per_window=5)
    logger.info(f"Saving window-level term tone to {TERM_WINDOW_TONE_CSV}")
    window_df.to_csv(TERM_WINDOW_TONE_CSV, index=False)

    shift_df = compute_cross_period_shifts(window_df)
    if shift_df.empty:
        logger.error("No valid early/late combinations to compute shifts. Aborting.")
        return

    logger.info(f"Saving cross-period shifts by speaker to {TERM_SHIFT_CSV}")
    shift_df.to_csv(TERM_SHIFT_CSV, index=False)

    final_df = aggregate_term_scores(
        shift_df,
        window_df,
        df_terms,
        significant_quantile=0.8
    )

    if final_df.empty:
        logger.warning("Final term score table is empty. Check data and thresholds.")
    else:
        logger.info(f"Saving final term score table to {TERM_FINAL_SCORE_CSV}")
        final_df.to_csv(TERM_FINAL_SCORE_CSV, index=False)

    # -----------------------------
    # ★ 直接更新 term_build_stats.json
    # 统计口径
    # - 结果集 final_df 是满足 early/late 覆盖度且可计算 tone shift 的 company-term
    # - 同时提供 unique term 的 unigram bigram 计数
    # -----------------------------
    try:
        patch: Dict[str, Any] = {}

        patch["term_tone_company_term_speaker_with_early_late"] = int(len(shift_df))

        if final_df is None or final_df.empty:
            patch["term_tone_company_term_total"] = 0
            patch["term_tone_unique_terms_total"] = 0
            patch["term_tone_unique_unigram_terms"] = 0
            patch["term_tone_unique_bigram_terms"] = 0

            patch["term_tone_significant_company_term_total"] = 0
            patch["term_tone_significant_unique_terms_total"] = 0
            patch["term_tone_significant_unique_unigram_terms"] = 0
            patch["term_tone_significant_unique_bigram_terms"] = 0
        else:
            patch["term_tone_company_term_total"] = int(len(final_df))

            all_terms = final_df["term"].astype(str).tolist()
            c_all = _count_unique_unigram_bigram(all_terms)
            patch["term_tone_unique_terms_total"] = c_all["unique_terms_total"]
            patch["term_tone_unique_unigram_terms"] = c_all["unique_unigram_terms"]
            patch["term_tone_unique_bigram_terms"] = c_all["unique_bigram_terms"]

            sig_df = final_df[final_df.get("is_significant", False) == True].copy()
            patch["term_tone_significant_company_term_total"] = int(len(sig_df))
            if sig_df.empty:
                patch["term_tone_significant_unique_terms_total"] = 0
                patch["term_tone_significant_unique_unigram_terms"] = 0
                patch["term_tone_significant_unique_bigram_terms"] = 0
            else:
                sig_terms = sig_df["term"].astype(str).tolist()
                c_sig = _count_unique_unigram_bigram(sig_terms)
                patch["term_tone_significant_unique_terms_total"] = c_sig["unique_terms_total"]
                patch["term_tone_significant_unique_unigram_terms"] = c_sig["unique_unigram_terms"]
                patch["term_tone_significant_unique_bigram_terms"] = c_sig["unique_bigram_terms"]

        update_json_inplace(TERM_BUILD_STATS_JSON, patch)
        logger.info(f"Appended Step 5.2 term stats into term_build_stats.json with keys {list(patch.keys())}")
    except Exception as e:
        logger.error(f"Failed to update term_build_stats.json in place: {e}")

    logger.info("===== Step 5.2: Term-level cross-period tone shift analysis DONE =====")


if __name__ == "__main__":
    main()
