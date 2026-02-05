# -*- coding: utf-8 -*-
"""
build_term_topic.py

RQ2 - Step 1: 构建“术语集合 ℓ_term”（基于原始文本 + 从金融/风险词典“学习”出来的约束）

目标：
- 找到“真正有金融含义的 n-gram”，用于后续 5.1 / 5.2：
  - 反映 speaker 在会议中反复提到、在意的 financial term；
  - 尽可能过滤掉：时间词、连接词、口水短语、计量词、情感形容词等。
- 本版本允许 1/2/3 元短语，后续可以按 n_tokens 再分类分析。
"""

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Set

import numpy as np
import pandas as pd

# -------------------------
# 配置
# -------------------------

# 原始文本根目录：默认 grasp_data/output
DEFAULT_RAW_ROOT_REL = Path("grasp_data") / "output"

# 术语筛选阈值（可以视情况调）
MIN_TERM_FREQ = 30      # 候选术语最小出现次数
MIN_DOC_COUNT = 5       # 至少出现的不同文档数

# 从词典“学习”核心 token 时的阈值
MIN_CORE_TOKEN_FREQ = 5     # 至少出现在多少个词典术语中
CORE_TOKEN_RATIO = 2.0      # financial_count >= ratio * risk_count 视为“金融核心 token”；反之为“风险核心 token”

# n-gram 中金融 anchor 占比下限
MIN_ANCHOR_RATIO = 0.5

# 打分时使用的特征及权重
SCORE_FEATURE_WEIGHTS = {
    "freq_log_scaled":        0.25,
    "doc_log_scaled":         0.20,
    "doc_entropy_scaled":     0.10,
    "fin_core_hits_scaled":   0.20,   # 命中金融核心 token 越多越好
    "financial_flag_scaled":  0.20,   # 出现在金融词典中加分
    "non_risk_flag_scaled":   0.05,   # 不靠近风险词典略微加分
}

# 基础停用词：连接词 / 代词 / 情态动词等
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "at", "by",
    "with", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "that", "this", "these", "those", "it", "its", "we", "you", "they", "he",
    "she", "him", "her", "them", "our", "their", "i", "me", "my", "your",
    "but", "not", "no", "so", "if", "then", "than", "about", "into", "out",
    "up", "down", "over", "under", "again", "further", "please", "pleased",
    # 高频助动词/情态动词/缩写
    "will", "would", "could", "should", "can", "cannot",
    "do", "does", "did",
    "ll", "re", "ve", "s", "d", "t",
    "going",
    # 典型口水动词/名词
    "result", "results",
    "continue", "continues", "continued", "continuing",
    "able",
    "lot", "lots",
    "end", "ending",
    "start", "starting",
    "begin", "began", "begun",
    "finish", "finished",
    "now", "today", "tonight", "yesterday", "tomorrow",
    "forward",
    "around",
    "kind", "kinds", "sort", "sorts",
    "thing", "things",
    "really", "very", "quite", "pretty",
    "much", "more", "less",
    "better", "worse",
    "first", "second", "third",
    "next", "last", "prior",
}

# “口水开头”词：如果 term 以这些开头，且不在金融词典中，极大概率是口水短语
BORING_STARTS = {
    "we", "i", "you", "they", "he", "she", "it",
    "this", "that", "these", "those",
    "as", "so", "but",
    "in", "on", "for", "at", "by", "with", "from",
    "a", "an", "the",
    "if", "then",
    "and", "or",
    "now", "today",
    "result", "results",
    "continue", "continued", "continues",
    "able", "lot", "lots",
}

# 纯时间类 token：单独出现时不算金融概念
BANNED_TIME_TOKENS = {
    "year", "years", "quarter", "quarters", "month", "months",
    "week", "weeks", "day", "days",
    "yoy", "qoq",
    "q1", "q2", "q3", "q4",
    "fy", "fiscal",
}

# 纯计量单位 token：单独出现时不算金融概念
BANNED_UNIT_TOKENS = {
    "million", "millions",
    "billion", "billions",
    "thousand", "thousands",
    "percent", "percentage",
    "bps", "bp",
}

# 比较/连词类 token
BORING_COMPARISON_TOKENS = {
    "compared", "versus", "vs", "relative", "against",
    "due", "because",
}

# 手工指定一批“明显金融”的 anchor token
MANUAL_FIN_TOKENS = {
    "cash", "revenue", "revenues", "sales", "turnover",
    "profit", "profits", "earnings", "ebit", "ebitda",
    "margin", "margins", "spread", "spreads",
    "income", "expense", "expenses", "cost", "costs",
    "capital", "capex", "opex", "dividend", "dividends",
    "loan", "loans", "credit", "credits", "facility", "facilities",
    "debt", "liquidity", "leverage", "covenant", "covenants",
    "impairment", "impairments", "provision", "provisions",
    "guidance", "outlook",
    "asset", "assets", "liability", "liabilities",
    "equity", "share", "shares", "stock", "stocks",
    "bond", "bonds",
    "rate", "rates", "interest", "yield", "yields",
    "hedge", "hedging", "derivative", "derivatives",
    "working", "capital",
}

# “明显金融”的词干
FINANCIAL_STEM_SUBSTRINGS = [
    "cash", "revenue", "revenues", "sale", "sales", "turnover",
    "earning", "earnings", "profit", "loss", "income",
    "margin", "spread", "cost", "expense", "expenses",
    "capital", "capex", "opex",
    "dividend", "dividends",
    "loan", "loans", "credit", "credits", "facility", "facilities",
    "debt", "liquidit", "liquidity",
    "equity", "asset", "assets", "liabilit", "liability", "liabilities",
    "bond", "bonds", "coupon", "coupons",
    "rate", "rates", "yield", "yields", "interest",
    "hedge", "hedging", "derivativ", "derivative", "derivatives",
    "default", "defaults", "delinquen", "delinquency",
    "covenant", "covenants",
    "impair", "impairment",
    "provision", "provisions",
    "writeoff", "write-off", "write-down", "writedown",
    "tax", "taxes", "taxation",
    "fund", "funding",
    "refinanc", "refinancing",
    "maturit", "maturity", "maturities",
    "guidance", "outlook",
    "working capital",
    "leasing", "lease", "leases",
    "option", "options", "swap", "swaps",
    "portfolio",
    "pension", "pensions",
    "collateral",
]


# -------------------------
# 工具函数
# -------------------------

def normalize_term(s: str) -> str:
    """术语规范化：小写、去首尾空格、内部空格压缩。"""
    return " ".join(s.lower().strip().split())


def compute_entropy(counts: Iterable[int]) -> float:
    """给定一组计数，计算 Shannon 熵。"""
    total = float(sum(counts))
    if total <= 0:
        return 0.0
    ent = 0.0
    for c in counts:
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p + 1e-12)
    return ent


def tokenize_english(text: str) -> List[str]:
    """
    简单英文 tokenizer：只取字母串。
    数字（2022、3.5）不会变成 token。
    """
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())


def term_has_clear_financial_stem(non_stop_tokens: List[str]) -> bool:
    """判断若干 non-stop token 中，是否至少有一个包含明显金融词干。"""
    if not non_stop_tokens:
        return False

    joined = " ".join(non_stop_tokens)
    for stem in FINANCIAL_STEM_SUBSTRINGS:
        if " " in stem:
            if stem in joined:
                return True

    for t in non_stop_tokens:
        for stem in FINANCIAL_STEM_SUBSTRINGS:
            if " " in stem:
                continue
            if stem in t:
                return True
    return False


def extract_ngrams(
    tokens: List[str],
    n: int,
    financial_anchor_tokens: Set[str],
) -> List[str]:
    """
    从 token 序列中抽取 n-gram 术语（支持 n=1/2/3）。

    规则：
    - span 里至少有一个非停用词；
    - non-stop token 中，至少一个在 financial_anchor_tokens 中；
    - 且 non-stop token 中，金融 anchor 占比 >= MIN_ANCHOR_RATIO；
    - 不能是纯时间 / 纯计量 / 纯比较短语；
    - 最终 term 会删除所有 STOPWORDS 和 BANNED_UNIT_TOKENS；
      若删完后 token 数 == 0，则丢弃。
      （★ 不再强制要求 token 数 ≥ 2，允许一元金融 term。）
    """
    terms: List[str] = []
    if len(tokens) < n:
        return terms

    for i in range(len(tokens) - n + 1):
        span = tokens[i: i + n]

        # span 全是停用词 → 丢
        if all(tok in STOPWORDS for tok in span):
            continue

        # span 内有意义的 token（不含 stopword）
        non_stop = [t for t in span if t not in STOPWORDS]
        if not non_stop:
            continue

        # 时间 / 计量 / 比较 兜底过滤
        if all(t in BANNED_TIME_TOKENS for t in non_stop):
            continue
        if all(t in BANNED_UNIT_TOKENS for t in non_stop):
            continue
        if all((t in BORING_COMPARISON_TOKENS) or (t in STOPWORDS) for t in non_stop):
            continue

        # anchor 约束
        anchor_hits = sum(1 for t in non_stop if t in financial_anchor_tokens)
        if anchor_hits == 0:
            continue
        if anchor_hits / float(len(non_stop)) < MIN_ANCHOR_RATIO:
            continue

        # 构造最终术语：删掉所有介词/连词/冠词 + 纯计量词
        content_tokens = [
            t for t in span
            if t not in STOPWORDS and t not in BANNED_UNIT_TOKENS
        ]

        # ★ 保证还有内容；允许只有一个 content token（即 1-gram）
        term = normalize_term(" ".join(content_tokens))
        if not term:
            continue

        terms.append(term)

    return terms


# -------------------------
# 词典加载 & 从词典学习核心 token
# -------------------------

def load_lexicon_map(path: Path) -> Dict[str, List[str]]:
    """
    通用词典加载：term_norm -> [category1, ...]
    兼容结构：
    1) {"catA": ["t1", "t2"], ...}
    2) ["t1", "t2", ...]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    term_to_cats: Dict[str, List[str]] = defaultdict(list)
    if isinstance(data, dict):
        for cat, terms in data.items():
            if not isinstance(terms, (list, tuple)):
                continue
            for t in terms:
                t_norm = normalize_term(str(t))
                if not t_norm:
                    continue
                term_to_cats[t_norm].append(cat)
    else:
        for t in data:
            t_norm = normalize_term(str(t))
            if not t_norm:
                continue
            term_to_cats[t_norm].append("unknown")
    return term_to_cats


def learn_core_tokens_from_lexicons(
    financial_lex_map: Dict[str, List[str]],
    risk_lex_map: Dict[str, List[str]],
    min_freq: int = MIN_CORE_TOKEN_FREQ,
    ratio: float = CORE_TOKEN_RATIO,
) -> Tuple[Dict[str, int], Dict[str, int], Set[str], Set[str]]:
    """从金融/风险词典自动学金融核心 token 与风险核心 token。"""
    fin_tok_counts: Dict[str, int] = defaultdict(int)
    risk_tok_counts: Dict[str, int] = defaultdict(int)

    for term in financial_lex_map.keys():
        toks = set(normalize_term(term).split())
        for tok in toks:
            if len(tok) < 3:
                continue
            fin_tok_counts[tok] += 1

    for term in risk_lex_map.keys():
        toks = set(normalize_term(term).split())
        for tok in toks:
            if len(tok) < 3:
                continue
            risk_tok_counts[tok] += 1

    fin_core_tokens: Set[str] = set()
    risk_core_tokens: Set[str] = set()

    all_tokens = set(fin_tok_counts.keys()) | set(risk_tok_counts.keys())
    for tok in all_tokens:
        f = fin_tok_counts.get(tok, 0)
        r = risk_tok_counts.get(tok, 0)
        if f >= min_freq and f >= ratio * max(r, 1):
            fin_core_tokens.add(tok)
        if r >= min_freq and r >= ratio * max(f, 1):
            risk_core_tokens.add(tok)

    print(f"[info] 从词典学习得到金融核心 token 数量：{len(fin_core_tokens)}")
    print(f"[info] 从词典学习得到风险核心 token 数量：{len(risk_core_tokens)}")

    return fin_tok_counts, risk_tok_counts, fin_core_tokens, risk_core_tokens


def is_potential_financial_token(tok: str) -> bool:
    """判断一个 token 是否有可能是金融 anchor。"""
    if len(tok) < 3:
        return False
    if tok in STOPWORDS or tok in BANNED_TIME_TOKENS or tok in BANNED_UNIT_TOKENS:
        return False
    for stem in FINANCIAL_STEM_SUBSTRINGS:
        if " " in stem:
            continue
        if stem in tok:
            return True
    return False


def build_financial_anchor_tokens(
    financial_lex_map: Dict[str, List[str]],
    fin_core_tokens: Set[str],
) -> Set[str]:
    """
    构造“金融 anchor token”集合：
    - 金融核心 token；
    - 金融词典中 term 的 token（只保留 is_potential_financial_token 为 True 的）；
    - 手工指定的 MANUAL_FIN_TOKENS。
    """
    anchor_tokens: Set[str] = set()

    for tok in fin_core_tokens:
        if is_potential_financial_token(tok):
            anchor_tokens.add(tok)

    for term in financial_lex_map.keys():
        toks = normalize_term(term).split()
        for tok in toks:
            if is_potential_financial_token(tok):
                anchor_tokens.add(tok)

    for t in MANUAL_FIN_TOKENS:
        for tok in normalize_term(t).split():
            if is_potential_financial_token(tok):
                anchor_tokens.add(tok)

    print(f"[info] 构造得到金融 anchor token 数量：{len(anchor_tokens)}")
    return anchor_tokens


# -------------------------
# 主流程：从原始文本构建术语统计
# -------------------------

def collect_terms_from_raw(
    raw_root: Path,
    financial_anchor_tokens: Set[str],
) -> Tuple[pd.DataFrame, int]:
    """
    遍历 grasp_data/output/Type_* 下所有 .txt 文件，
    抽取 1/2/3-gram 术语，形成“术语 × 文档”长表。

    返回：
        long_df: term × doc 长表
        total_tokens: 原始文本 token 总数
    """
    file_paths = sorted(raw_root.glob("Type_*/*.txt"))
    if not file_paths:
        raise RuntimeError(f"在 {raw_root} 下未找到任何 Type_*/*.txt 文件，请检查路径。")

    print(f"[info] 在 {raw_root} 下找到原始文本文件数：{len(file_paths)}")

    records = []
    total_tokens = 0

    for idx, path in enumerate(file_paths, start=1):
        doc_id = str(path.relative_to(raw_root))
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[warn] 读取文件失败，跳过: {path}，原因: {e}")
            continue

        tokens = tokenize_english(text)
        total_tokens += len(tokens)

        if idx <= 3:
            print(f"[debug] 示例文档 {idx}: {doc_id}")
            print(f"        前 20 个 token: {tokens[:20]}")

        # ★ 现在同时抽取 1-gram / 2-gram / 3-gram
        terms_1 = extract_ngrams(tokens, 1, financial_anchor_tokens)
        terms_2 = extract_ngrams(tokens, 2, financial_anchor_tokens)
        terms_3 = extract_ngrams(tokens, 3, financial_anchor_tokens)
        all_terms = terms_1 + terms_2 + terms_3

        if not all_terms:
            continue

        cnt = Counter(all_terms)
        for term, c in cnt.items():
            records.append({"term": term, "doc_id": doc_id, "count": int(c)})

        if idx % 100 == 0:
            print(f"[info] 已处理文档 {idx}/{len(file_paths)}")

    print(f"[info] 原始文本总 token 数约为: {total_tokens}")

    long_df = pd.DataFrame.from_records(records)
    if long_df.empty:
        raise RuntimeError("从原始文本中未抽取到任何术语，请检查 tokenizer 或 anchor 设置。")

    return long_df, total_tokens


def build_term_features_from_long(
    long_df: pd.DataFrame,
    financial_lex_map: Dict[str, List[str]],
    risk_lex_map: Dict[str, List[str]],
    fin_core_tokens: Set[str],
    risk_core_tokens: Set[str],
) -> pd.DataFrame:
    """汇总构建术语画像特征。"""
    print("[info] 开始按术语汇总特征...")

    g = long_df.groupby("term")

    term_df = pd.DataFrame(index=g.size().index)
    term_df["term"] = term_df.index
    term_df["freq_total"] = g["count"].sum().astype(int)
    term_df["n_docs"] = g["doc_id"].nunique().astype(int)

    doc_counts = g["doc_id"].value_counts()
    doc_entropy = {}
    for term, sub in doc_counts.groupby(level=0):
        counts = sub.values
        doc_entropy[term] = compute_entropy(counts)
    term_df["doc_entropy"] = term_df["term"].map(doc_entropy).astype(float)

    # 金融词典 membership
    def map_financial(term: str):
        cats = financial_lex_map.get(term, [])
        return len(cats) > 0, ";".join(sorted(set(cats))) if cats else ""

    fin_flags = term_df["term"].apply(map_financial)
    term_df["in_financial_lexicon"] = [flag for flag, _ in fin_flags]
    term_df["financial_lexicon_categories"] = [cats for _, cats in fin_flags]
    term_df["financial_cat_count"] = term_df["financial_lexicon_categories"].apply(
        lambda s: 0 if not s else len(s.split(";"))
    )
    term_df["in_financial_flag"] = term_df["in_financial_lexicon"].astype(int)

    # 风险词典 membership
    def map_risk(term: str):
        cats = risk_lex_map.get(term, [])
        return len(cats) > 0, ";".join(sorted(set(cats))) if cats else ""

    risk_flags = term_df["term"].apply(map_risk)
    term_df["in_risk_lexicon"] = [flag for flag, _ in risk_flags]
    term_df["risk_lexicon_categories"] = [cats for _, cats in risk_flags]
    term_df["risk_cat_count"] = term_df["risk_lexicon_categories"].apply(
        lambda s: 0 if not s else len(s.split(";"))
    )
    term_df["in_risk_flag"] = term_df["in_risk_lexicon"].astype(int)

    # 核心 token 命中数
    def count_core_hits(term: str, core_tokens: Set[str]) -> int:
        toks = set(term.split())
        return sum(1 for t in toks if t in core_tokens)

    term_df["fin_core_hits"] = term_df["term"].apply(
        lambda t: count_core_hits(t, fin_core_tokens)
    ).astype(int)
    term_df["risk_core_hits"] = term_df["term"].apply(
        lambda t: count_core_hits(t, risk_core_tokens)
    ).astype(int)

    term_df.reset_index(drop=True, inplace=True)
    return term_df


def is_boring_phrase(term: str, in_financial_flag: int) -> bool:
    """
    判断一个 term 是否“明显没有金融含义”的口水短语/时间短语/计量短语/比较短语。
    只有在该 term 又不在金融词典中时，这些规则才生效。
    """
    toks = term.split()
    if not toks:
        return True

    # non_stop：去掉 stopword + 纯计量词
    non_stop = [t for t in toks if t not in STOPWORDS and t not in BANNED_UNIT_TOKENS]
    if not non_stop:
        return True

    # term 不在金融 lexicon 且没有任何金融词干 → 直接 boring
    if in_financial_flag == 0 and not term_has_clear_financial_stem(non_stop):
        return True

    if toks[0] in BORING_STARTS and in_financial_flag == 0:
        return True

    if all(t in BANNED_TIME_TOKENS for t in non_stop) and in_financial_flag == 0:
        return True

    if all(t in BANNED_UNIT_TOKENS for t in non_stop) and in_financial_flag == 0:
        return True

    if all((t in BORING_COMPARISON_TOKENS) or (t in STOPWORDS) for t in non_stop) and in_financial_flag == 0:
        return True

    return False


def filter_and_score_terms(term_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    基础过滤 + 金融/风险约束 + 口水过滤 + 打分。

    返回：
        filtered: 过滤并打分后的术语表（包含 1/2/3-gram）
        stats:    若干汇总统计（术语总数 / 基础过滤后数量 / 最终保留数量）
    """
    print("[info] 对术语进行基础过滤与多指标无监督打分...")

    term_df = term_df.copy()

    # ★ 保留 n_tokens 信息，方便后续按 1/2/3-gram 分析，但不再用来过滤掉 1-gram
    term_df["n_tokens"] = term_df["term"].str.split().str.len().fillna(0).astype(int)

    term_df["first_tok"] = term_df["term"].str.split().str[0].fillna("")
    term_df["is_boring"] = term_df.apply(
        lambda row: is_boring_phrase(row["term"], int(row["in_financial_flag"])),
        axis=1,
    )

    mask_basic = (
        (term_df["freq_total"] >= MIN_TERM_FREQ)
        & (term_df["n_docs"] >= MIN_DOC_COUNT)
    )

    mask_fin_core = (term_df["fin_core_hits"] > 0)
    mask_fin_lex = (term_df["in_financial_flag"] == 1)
    mask_financial = mask_fin_core | mask_fin_lex

    mask_not_risk = (term_df["in_risk_flag"] == 0) & (term_df["risk_core_hits"] == 0)
    mask_not_boring = (term_df["is_boring"] == 0)

    mask_all = mask_basic & mask_financial & mask_not_risk & mask_not_boring
    filtered = term_df.loc[mask_all].copy()

    total_terms = int(len(term_df))
    basic_count = int(mask_basic.sum())

    print(
        f"[info] 术语总数：{total_terms}；"
        f"基础过滤后：{basic_count}；"
        f"金融相关且非风险（去掉口水/年份/计量）后：{len(filtered)}"
    )

    if filtered.empty:
        print("[warn] 过滤后术语为空，请检查阈值 / 规则设置。")
        filtered = term_df.copy()

    filtered["freq_log"] = np.log1p(filtered["freq_total"])
    filtered["doc_log"] = np.log1p(filtered["n_docs"])
    filtered["financial_flag_scaled"] = filtered["in_financial_flag"].astype(float)
    filtered["non_risk_flag_scaled"] = (
        ((filtered["in_risk_flag"] == 0) & (filtered["risk_core_hits"] == 0))
        .astype(float)
    )
    filtered["fin_core_hits"] = filtered["fin_core_hits"].astype(float)

    to_scale = {
        "freq_log": "freq_log_scaled",
        "doc_log": "doc_log_scaled",
        "doc_entropy": "doc_entropy_scaled",
        "fin_core_hits": "fin_core_hits_scaled",
    }

    for src, dst in to_scale.items():
        col = filtered[src].astype(float)
        c_min, c_max = col.min(), col.max()
        if c_max > c_min:
            filtered[dst] = (col - c_min) / (c_max - c_min)
        else:
            filtered[dst] = 0.0

    score = np.zeros(len(filtered), dtype=float)
    for feat, w in SCORE_FEATURE_WEIGHTS.items():
        if feat not in filtered.columns:
            continue
        score += w * filtered[feat].values
    filtered["score"] = score

    filtered.sort_values("score", ascending=False, inplace=True)
    filtered.reset_index(drop=True, inplace=True)

    final_count = int(len(filtered))
    stats = {
        "term_total": total_terms,
        "basic_filtered": basic_count,
        "financial_nonrisk_filtered": final_count,
    }

    return filtered, stats


# -------------------------
# CLI 主入口
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="RQ2 - 基于原始文本 + 从金融/风险词典学习的约束构建术语集合 ℓ_term（支持 1/2/3-gram）"
    )

    this_file = Path(__file__).resolve()
    proj_root = this_file.parents[1]

    default_raw_root = proj_root / DEFAULT_RAW_ROOT_REL
    default_financial_lex_path = proj_root / "wordbag" / "financial_lexicon.json"
    default_risk_lex_path = proj_root / "wordbag" / "risk_lexicons.json"
    default_out_dir = proj_root / "term_analyse" / "output"

    parser.add_argument(
        "--raw-root",
        type=str,
        default=str(default_raw_root),
        help="原始文本根目录（默认：project/grasp_data/output）",
    )
    parser.add_argument(
        "--financial-lexicon-path",
        type=str,
        default=str(default_financial_lex_path),
        help="金融词典 JSON 文件路径（默认：project/wordbag/financial_lexicon.json）",
    )
    parser.add_argument(
        "--risk-lexicon-path",
        type=str,
        default=str(default_risk_lex_path),
        help="风险词典 JSON 文件路径（默认：project/wordbag/risk_lexicons.json）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(default_out_dir),
        help="输出目录（默认：project/term_analyse/output）",
    )

    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    financial_lex_path = Path(args.financial_lexicon_path)
    risk_lex_path = Path(args.risk_lexicon_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] 项目根目录: {proj_root}")
    print(f"[info] 原始文本根目录: {raw_root}")
    print(f"[info] 金融词典文件: {financial_lex_path}")
    print(f"[info] 风险词典文件: {risk_lex_path}")
    print(f"[info] 输出目录: {output_dir}")

    # 1. 词典 + 核心 token
    financial_lex_map = load_lexicon_map(financial_lex_path)
    risk_lex_map = load_lexicon_map(risk_lex_path)
    print(f"[info] 金融词典术语数量（去重）：{len(financial_lex_map)}")
    print(f"[info] 风险词典术语数量（去重）：{len(risk_lex_map)}")

    fin_tok_counts, risk_tok_counts, fin_core_tokens, risk_core_tokens = \
        learn_core_tokens_from_lexicons(financial_lex_map, risk_lex_map)

    # 1.5 anchor 集
    financial_anchor_tokens = build_financial_anchor_tokens(
        financial_lex_map=financial_lex_map,
        fin_core_tokens=fin_core_tokens,
    )

    # 2. 从原始文本收集术语长表（含 1/2/3-gram）
    long_df, total_tokens = collect_terms_from_raw(raw_root, financial_anchor_tokens)
    print(f"[info] 术语长表形状: {long_df.shape}")

    # 3. 构建术语画像
    term_df = build_term_features_from_long(
        long_df=long_df,
        financial_lex_map=financial_lex_map,
        risk_lex_map=risk_lex_map,
        fin_core_tokens=fin_core_tokens,
        risk_core_tokens=risk_core_tokens,
    )

    # 4. 筛选 + 打分
    term_scored, filter_stats = filter_and_score_terms(term_df)

    # 5. 输出
    term_features_path = output_dir / "term_features.parquet"
    terms_selected_csv = output_dir / "terms_selected_from_raw.csv"
    terms_selected_json = output_dir / "terms_selected_from_raw.json"

    term_df.to_parquet(term_features_path, index=False)
    print(f"[info] 已保存所有术语特征到: {term_features_path}")

    term_scored.to_csv(terms_selected_csv, index=False)
    print(f"[info] 已保存打分后术语表到: {terms_selected_csv}")
    print(f"[info] 打分后术语数量: {len(term_scored)}")

    selected_terms = term_scored["term"].tolist()
    print(f"[info] 将 {len(selected_terms)} 个术语写入 JSON（与 CSV 一致）")
    with open(terms_selected_json, "w", encoding="utf-8") as f:
        json.dump(selected_terms, f, ensure_ascii=False, indent=2)

    print(f"[info] 已保存术语集合 ℓ_term 到: {terms_selected_json}")

    # 6. 保存四个关键统计，方便后续 5.2 / 5.3 调用
    stats_json_path = output_dir / "term_build_stats.json"
    stats_payload = {
        "total_tokens": int(total_tokens),
        "term_total": int(filter_stats.get("term_total", len(term_df))),
        "term_after_basic_filter": int(filter_stats.get("basic_filtered", 0)),
        "term_after_financial_nonrisk_filter": int(
            filter_stats.get("financial_nonrisk_filtered", len(term_scored))
        ),
    }
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats_payload, f, ensure_ascii=False, indent=2)
    print(f"[info] 已保存术语构建统计到: {stats_json_path}")

    print("[info] build_term_topic.py 完成。")


if __name__ == "__main__":
    main()
