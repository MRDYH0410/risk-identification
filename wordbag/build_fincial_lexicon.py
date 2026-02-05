# -*- coding: utf-8 -*-
"""
build_fincial_lexicon.py   （按需求拼写）

基于：
  - wordbag/finance_standard_wordbag/Loughran-McDonald_MasterDictionary_1993-2024.csv
  - wordbag/finance_standard_wordbag/Sentences_50Agree.txt
  - wordbag/finance_standard_wordbag/Sentences_66Agree.txt
  - wordbag/finance_standard_wordbag/Sentences_75Agree.txt
  - wordbag/finance_standard_wordbag/Sentences_AllAgree.txt

构建统一的金融词袋 JSON:
  project/wordbag/financial_lexicon.json

JSON 结构示例：
{
  "lm_general": [...],
  "lm_risk_negative": [...],
  "lm_risk_uncertainty": [...],
  "lm_risk_litigious": [...],
  "lm_risk_constraining": [...],
  "lm_risk_strong_modal": [...],
  "lm_risk_weak_modal": [...],
  "phrasebank_unigram": [...],
  "phrasebank_bigram": [...],
  "phrasebank_trigram": [...]
}
"""

import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Set

import pandas as pd

# -------------------------
# 基础配置（可按需调整）
# -------------------------

# PhraseBank n-gram 频率阈值
UNIGRAM_MIN_FREQ = 5
BIGRAM_MIN_FREQ = 5
TRIGRAM_MIN_FREQ = 3

# 粗略英文 stopwords（只用于过滤“纯停用词” n-gram）
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "at", "by",
    "with", "from", "as", "is", "are", "was", "were", "be", "been", "being",
    "that", "this", "these", "those", "it", "its", "we", "you", "they", "he",
    "she", "him", "her", "them", "our", "their", "i", "me", "my", "your",
    "but", "not", "no", "so", "if", "then", "than", "about", "into", "out",
    "up", "down", "over", "under", "again", "further"
}


# -------------------------
# 工具函数
# -------------------------

def normalize_term(s: str) -> str:
    """术语规范化：小写、去首尾空格、内部空格压缩。"""
    return " ".join(s.lower().strip().split())


def tokenize_english(text: str) -> List[str]:
    """
    非常简单的英文 tokenizer：
    - 提取由字母开头、允许中间出现 ' 或 - 的片段；
    - 全部转小写。
    """
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())


def extract_ngrams(tokens: List[str], n: int) -> List[str]:
    """
    从 token 序列中抽取 n-gram。
    规则：
    - 至少一个 token 不是 stopword；
    - 术语字符串规范化为小写 + 单空格。
    """
    terms: List[str] = []
    if len(tokens) < n:
        return terms
    for i in range(len(tokens) - n + 1):
        span = tokens[i: i + n]
        if all(tok in STOPWORDS for tok in span):
            continue
        term = normalize_term(" ".join(span))
        if term:
            terms.append(term)
    return terms


def find_col(df: pd.DataFrame, substrings) -> str:
    """
    在 df.columns 中寻找包含某些子串的列名（忽略大小写），返回第一个匹配的列。
    如果找不到，返回 None。
    """
    subs = [s.lower() for s in substrings]
    for col in df.columns:
        cl = col.lower()
        for s in subs:
            if s in cl:
                return col
    return None


# -------------------------
# 1) 处理 Loughran-McDonald 词典
# -------------------------

def build_lm_lexicon(lm_csv_path: Path) -> Dict[str, List[str]]:
    """
    从 Loughran-McDonald 主词典中构建若干类别：
      - lm_general: 非风险类的一般金融词
      - lm_risk_negative / lm_risk_uncertainty / lm_risk_litigious
      - lm_risk_constraining / lm_risk_strong_modal / lm_risk_weak_modal
    """
    print(f"[info] 读取 LM 词典: {lm_csv_path}")
    df = pd.read_csv(lm_csv_path)

    # 标准列名映射（不放心可以打印 df.columns 看一下）
    word_col = find_col(df, ["word"])
    if word_col is None:
        raise RuntimeError("在 LM 词典中找不到 'word' 列，请检查 CSV 结构。")

    neg_col = find_col(df, ["negative"])
    unc_col = find_col(df, ["uncertainty"])
    lit_col = find_col(df, ["litigious"])
    con_col = find_col(df, ["constraining"])
    strong_col = find_col(df, ["strongmodal", "strong modal"])
    weak_col = find_col(df, ["weakmodal", "weak modal"])

    print("[info] LM 列匹配情况：")
    print(f"   word_col       = {word_col}")
    print(f"   negative_col   = {neg_col}")
    print(f"   uncertainty_col= {unc_col}")
    print(f"   litigious_col  = {lit_col}")
    print(f"   constraining_col= {con_col}")
    print(f"   strong_modal_col= {strong_col}")
    print(f"   weak_modal_col  = {weak_col}")

    lm_general: Set[str] = set()
    lm_neg: Set[str] = set()
    lm_unc: Set[str] = set()
    lm_lit: Set[str] = set()
    lm_con: Set[str] = set()
    lm_smod: Set[str] = set()
    lm_wmod: Set[str] = set()

    for _, row in df.iterrows():
        w_raw = str(row[word_col])
        w = normalize_term(w_raw)
        # 只保留纯字母词，且长度>1
        if not w or not w.replace("-", "").replace("'", "").isalpha():
            continue
        if len(w) <= 1:
            continue

        is_risk_like = False

        def flag(col_name, target_set: Set[str]):
            nonlocal is_risk_like
            if col_name is None:
                return
            try:
                v = row[col_name]
            except KeyError:
                return
            # LM 中这些列通常是 0/1 或计数，只要 >0 就认为有该属性
            try:
                if float(v) > 0:
                    target_set.add(w)
                    is_risk_like = True
            except Exception:
                pass

        flag(neg_col, lm_neg)
        flag(unc_col, lm_unc)
        flag(lit_col, lm_lit)
        flag(con_col, lm_con)
        flag(strong_col, lm_smod)
        flag(weak_col, lm_wmod)

        # 一般金融词：没有被标记为任何风险/情绪类别的词
        if not is_risk_like:
            lm_general.add(w)

    print(f"[info] LM 一般金融词 lm_general 数量: {len(lm_general)}")
    print(f"[info] LM 风险相关词：")
    print(f"       negative     : {len(lm_neg)}")
    print(f"       uncertainty  : {len(lm_unc)}")
    print(f"       litigious    : {len(lm_lit)}")
    print(f"       constraining : {len(lm_con)}")
    print(f"       strong_modal : {len(lm_smod)}")
    print(f"       weak_modal   : {len(lm_wmod)}")

    lm_lexicon = {
        "lm_general": sorted(lm_general),
        "lm_risk_negative": sorted(lm_neg),
        "lm_risk_uncertainty": sorted(lm_unc),
        "lm_risk_litigious": sorted(lm_lit),
        "lm_risk_constraining": sorted(lm_con),
        "lm_risk_strong_modal": sorted(lm_smod),
        "lm_risk_weak_modal": sorted(lm_wmod),
    }
    return lm_lexicon


# -------------------------
# 2) 处理 Financial PhraseBank
# -------------------------

def parse_phrasebank_file(path: Path):
    """
    逐行读取 PhraseBank 句子文件，格式通常是：
      <sentence>@<label>
    这里只要 sentence。
    """
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 按最后一个 @ 切分，避免句子内部的 @ 干扰
            parts = line.rsplit("@", 1)
            if len(parts) == 2:
                sent, label = parts
            else:
                sent, label = line, ""
            yield sent.strip(), label.strip()


def build_phrasebank_lexicon(pb_dir: Path) -> Dict[str, List[str]]:
    """
    从 finance_standard_wordbag 中的 Sentences_*Agree.txt 抽取
    unigram / bigram / trigram 作为金融短语词袋。
    """
    paths = sorted(pb_dir.glob("Sentences_*Agree.txt"))
    if not paths:
        raise RuntimeError(f"在 {pb_dir} 下未找到 Sentences_*Agree.txt")

    print("[info] 使用 PhraseBank 文件：")
    for p in paths:
        print("   -", p.name)

    cnt_uni = Counter()
    cnt_bi = Counter()
    cnt_tri = Counter()

    n_sent = 0

    for path in paths:
        for sent, label in parse_phrasebank_file(path):
            n_sent += 1
            toks = tokenize_english(sent)
            # unigram
            for tok in toks:
                if tok in STOPWORDS:
                    continue
                cnt_uni[tok] += 1
            # bigram / trigram
            for term in extract_ngrams(toks, 2):
                cnt_bi[term] += 1
            for term in extract_ngrams(toks, 3):
                cnt_tri[term] += 1

    print(f"[info] PhraseBank 总句子数: {n_sent}")
    print(f"[info] PhraseBank unigram 数量: {len(cnt_uni)}")
    print(f"[info] PhraseBank bigram 数量: {len(cnt_bi)}")
    print(f"[info] PhraseBank trigram 数量: {len(cnt_tri)}")

    uni_terms = [t for t, c in cnt_uni.items() if c >= UNIGRAM_MIN_FREQ]
    bi_terms = [t for t, c in cnt_bi.items() if c >= BIGRAM_MIN_FREQ]
    tri_terms = [t for t, c in cnt_tri.items() if c >= TRIGRAM_MIN_FREQ]

    uni_terms = sorted(uni_terms, key=lambda t: (-cnt_uni[t], t))
    bi_terms = sorted(bi_terms, key=lambda t: (-cnt_bi[t], t))
    tri_terms = sorted(tri_terms, key=lambda t: (-cnt_tri[t], t))

    print(f"[info] 过滤后 unigram(>= {UNIGRAM_MIN_FREQ}): {len(uni_terms)}")
    print(f"[info] 过滤后 bigram (>= {BIGRAM_MIN_FREQ}): {len(bi_terms)}")
    print(f"[info] 过滤后 trigram(>= {TRIGRAM_MIN_FREQ}): {len(tri_terms)}")

    # 打印几个例子看看
    print("[info] PhraseBank unigram Top 20:")
    for t in uni_terms[:20]:
        print(f"   {t}  (freq={cnt_uni[t]})")

    print("[info] PhraseBank bigram Top 20:")
    for t in bi_terms[:20]:
        print(f"   {t}  (freq={cnt_bi[t]})")

    print("[info] PhraseBank trigram Top 20:")
    for t in tri_terms[:20]:
        print(f"   {t}  (freq={cnt_tri[t]})")

    pb_lexicon = {
        "phrasebank_unigram": uni_terms,
        "phrasebank_bigram": bi_terms,
        "phrasebank_trigram": tri_terms,
    }
    return pb_lexicon


# -------------------------
# 主入口
# -------------------------

def main():
    this_file = Path(__file__).resolve()
    proj_root = this_file.parents[1]

    finance_dir = proj_root / "wordbag" / "finance_standard_wordbag"
    lm_csv = finance_dir / "Loughran-McDonald_MasterDictionary_1993-2024.csv"
    out_path = proj_root / "wordbag" / "financial_lexicon.json"

    print(f"[info] 项目根目录: {proj_root}")
    print(f"[info] 标准金融词源目录: {finance_dir}")

    if not lm_csv.exists():
        raise RuntimeError(f"找不到 LM 词典 CSV: {lm_csv}")

    # 1) LM 词典 -> 金融 & 风险类别
    lm_lexicon = build_lm_lexicon(lm_csv)

    # 2) PhraseBank 句子 -> 金融短语 n-gram
    pb_lexicon = build_phrasebank_lexicon(finance_dir)

    # 3) 合并两个词典为一个 JSON
    merged = {}
    merged.update(lm_lexicon)
    merged.update(pb_lexicon)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f"[info] 已写出统一金融词袋到: {out_path}")


if __name__ == "__main__":
    main()
