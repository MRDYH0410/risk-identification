# -*- coding: utf-8 -*-
"""
calculate_distance.py —— 逐文件句子级风险距离（SCRisk 风格）

输入:
  <proj_root>/grasp_data/output/Type_1/*.txt
  <proj_root>/grasp_data/output/Type_2/*.txt
  <proj_root>/grasp_data/output/Type_3/*.txt

输出(逐文件一一对应):
  <proj_root>/wordbag/output/Type_k/<原文件名>.sentences.csv
  列: sent_idx, text(原样), token_len, sc_near_risk_count, sc_near_risk_ratio
"""

import os
import re
import json
import glob
from typing import List, Tuple, Dict, Sequence, DefaultDict
from collections import defaultdict

import pandas as pd

# ===================== 基础：不破坏原文的断句 =====================

HEADER_KEYS = [
    r"\bparticipants\b",
    r"\bcorporate participants\b",
    r"\bconference call participants\b",
    r"\boperator instructions\b",
    r"\bpresentation\b",
    r"\bprepared remarks\b",
    r"\bmanagement discussion section\b",
    r"\bquestion(?:\s+and)?\s+answer\b",
    r"\bq\s*&\s*a\b",
    r"\bsafe\s*harbor\b",
    r"\bforward[- ]looking statements\b",
    r"\bcopyright\b",
]
HEADER_BLOCK_OPEN = re.compile("|".join(HEADER_KEYS), flags=re.I)
OP_INSTR = re.compile(r"\(operator instructions\)", flags=re.I)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def drop_header_sections(raw: str) -> str:
    txt = raw.replace("\r\n", "\n").replace("\r", "\n")
    txt = OP_INSTR.sub(" ", txt)
    paras = re.split(r"\n\s*\n", txt)
    kept = []
    for p in paras:
        pl = p.lower().strip()
        if not pl:
            kept.append(p); continue
        if HEADER_BLOCK_OPEN.search(pl):
            continue
        if re.fullmatch(r"\d{2}:\d{2}:\d{2}", pl):
            continue
        if len(p.strip()) <= 60 and p.strip().isupper():
            continue
        kept.append(p)
    return "\n\n".join(kept)

def sentence_spans(text: str) -> List[Tuple[int, int]]:
    if not text:
        return []
    end_pat = re.compile(r"[\.!\?。！？]")
    spans, last = [], 0
    for m in end_pat.finditer(text):
        j = m.end()
        while j < len(text) and text[j] in ['"', "'", "”", "’", "）", ")", "»", "›", "]"]:
            j += 1
        k = j
        while k < len(text) and text[k].isspace():
            k += 1
        spans.append((last, k))
        last = k
    if last < len(text):
        spans.append((last, len(text)))
    return spans

def extract_sentences_preserving_text(raw: str) -> List[str]:
    body = drop_header_sections(raw)
    spans = sentence_spans(body)
    sents = [body[s:e] for s, e in spans]
    return [s for s in sents if s and not s.isspace()]

# ===================== 度量用分词（与词典一致） =====================

TOKEN_PAT = re.compile(r"[a-z0-9\-]+")

def norm_for_metric(s: str) -> str:
    return s.lower().replace("\u3000", " ").replace("\xa0", " ")

def tokenize_for_metric(s: str) -> List[str]:
    return TOKEN_PAT.findall(norm_for_metric(s))

# ===================== 新增：短句合并 =====================

def is_short_sentence(text: str, min_tokens: int, min_alnum_chars: int) -> bool:
    """满足任一条件即认为是短句：token数 < min_tokens 或 字母数字字符数 < min_alnum_chars。"""
    toks = tokenize_for_metric(text)
    if len(toks) < min_tokens:
        return True
    alnum_chars = len("".join(re.findall(r"[A-Za-z0-9]", text)))
    return alnum_chars < min_alnum_chars

def merge_short_sentences(sents: List[str],
                          min_tokens: int = 3,
                          min_alnum_chars: int = 10) -> List[str]:
    """
    把过短的句子并到相邻句子（优先并到前一句；若为首句则并到后一句）。
    保留原文字符（含空白），只是把字符串拼接。
    """
    if not sents:
        return sents[:]

    out: List[str] = []
    i = 0
    n = len(sents)

    while i < n:
        s = sents[i]
        if not is_short_sentence(s, min_tokens, min_alnum_chars):
            out.append(s)
            i += 1
            continue

        # 短句：优先并到前一句
        if out:
            out[-1] = out[-1] + s  # 直接拼接，保留原样空白
            i += 1
            # 合并后的结果如果仍然过短，继续把后面的短句拉进来，直到达到阈值或遇到长句
            while i < n and is_short_sentence(out[-1], min_tokens, min_alnum_chars):
                out[-1] = out[-1] + sents[i]
                i += 1
        else:
            # 首句且短：并到下一句
            if i + 1 < n:
                merged = s + sents[i + 1]
                i += 2
                # 如果合并后仍短，继续吸收后续短句
                while i < n and is_short_sentence(merged, min_tokens, min_alnum_chars):
                    merged = merged + sents[i]
                    i += 1
                out.append(merged)
            else:
                # 只有一个短句，直接放入
                out.append(s)
                i += 1

    return out

# ===================== 词典加载：支持 n-gram 短语 =====================

TermSeq = Tuple[str, ...]
IndexMap = Dict[str, List[TermSeq]]  # 首 token -> 可能的序列列表

def load_term_sequences(json_path: str) -> Tuple[List[TermSeq], List[TermSeq]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data: Dict[str, List[str]] = json.load(f)

    supply, risk = [], []
    for bucket, terms in data.items():
        b = (bucket or "").strip().lower()
        target = supply if b.startswith("flr_") else risk if b.startswith("lm_") else None
        if target is None:
            continue
        for t in terms:
            toks = tokenize_for_metric(t)
            if toks:
                target.append(tuple(toks))
    supply = sorted(set(supply))
    risk = sorted(set(risk))
    return supply, risk

def build_first_token_index(term_seqs: Sequence[TermSeq]) -> IndexMap:
    idx: DefaultDict[str, List[TermSeq]] = defaultdict(list)
    for seq in term_seqs:
        idx[seq[0]].append(seq)
    for k in idx:
        idx[k].sort(key=len, reverse=True)  # 先匹配更长短语
    return idx

# ===================== 在句子中查找短语起始位置 =====================

def find_term_positions(tokens: List[str], first_idx: IndexMap) -> List[int]:
    n = len(tokens)
    hits: List[int] = []
    for i in range(n):
        cand = first_idx.get(tokens[i])
        if not cand:
            continue
        for seq in cand:
            m = len(seq)
            if i + m > n:
                continue
            if tokens[i:i+m] == list(seq):
                hits.append(i)
                break   # 同起点命中最长即可
    return hits

# ===================== SCRisk 风格计数 =====================

def scrisk_like_count(tokens: List[str],
                      supply_pos: List[int],
                      risk_pos: List[int],
                      window: int = 10) -> Tuple[int, float]:
    n = len(tokens)
    if n == 0 or not supply_pos or not risk_pos:
        return 0, 0.0
    risk_pos_sorted = sorted(risk_pos)
    count = 0
    j = 0
    for i in sorted(supply_pos):
        while j + 1 < len(risk_pos_sorted) and abs(risk_pos_sorted[j + 1] - i) <= abs(risk_pos_sorted[j] - i):
            j += 1
        if abs(risk_pos_sorted[j] - i) <= window:
            count += 1
    return count, count / float(n)

# ===================== 逐文件 → 逐句 → 输出 =====================

def process_one_file(in_path: str,
                     out_path: str,
                     supply_idx: IndexMap,
                     risk_idx: IndexMap,
                     window: int = 10,
                     merge_short: bool = True,
                     min_tokens: int = 3,
                     min_chars: int = 10) -> None:
    raw = read_text(in_path)
    sentences = extract_sentences_preserving_text(raw)  # 原样句子

    if merge_short:
        sentences = merge_short_sentences(sentences, min_tokens, min_chars)

    rows = []
    for idx, s in enumerate(sentences):
        toks = tokenize_for_metric(s)
        sp = find_term_positions(toks, supply_idx)
        rp = find_term_positions(toks, risk_idx)
        c, r = scrisk_like_count(toks, sp, rp, window=window)
        rows.append({
            "sent_idx": idx,
            "text": s,                      # 原样（若合并则为原样拼接）
            "token_len": len(toks),
            "sc_near_risk_count": c,
            "sc_near_risk_ratio": round(r, 6)
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"[OK] {os.path.basename(in_path)} -> {out_path}  sents={len(rows)}")

def process_type_dir(type_in_dir: str,
                     type_out_dir: str,
                     lex_json: str,
                     window: int = 10,
                     merge_short: bool = True,
                     min_tokens: int = 3,
                     min_chars: int = 10) -> None:
    paths = sorted(glob.glob(os.path.join(type_in_dir, "*.txt")))
    if not paths:
        print(f"[WARN] empty: {type_in_dir}")
        return

    supply_seqs, risk_seqs = load_term_sequences(lex_json)
    supply_idx = build_first_token_index(supply_seqs)
    risk_idx   = build_first_token_index(risk_seqs)
    print(f"[Lexicon] supply_terms={len(supply_seqs)}  risk_terms={len(risk_seqs)}")

    for fp in paths:
        base = os.path.basename(fp)
        out_fp = os.path.join(type_out_dir, base + ".sentences.csv")
        process_one_file(fp, out_fp, supply_idx, risk_idx,
                         window=window,
                         merge_short=merge_short,
                         min_tokens=min_tokens,
                         min_chars=min_chars)

def main():
    base_dir = os.path.dirname(__file__)
    proj_root = os.path.abspath(os.path.join(base_dir, ".."))

    lex_json = os.path.join(proj_root, "wordbag", "risk_lexicons.json")

    t1_in = os.path.join(proj_root, "grasp_data", "output", "Type_1")
    t2_in = os.path.join(proj_root, "grasp_data", "output", "Type_2")
    t3_in = os.path.join(proj_root, "grasp_data", "output", "Type_3")

    t1_out = os.path.join(proj_root, "wordbag", "output", "Type_1")
    t2_out = os.path.join(proj_root, "wordbag", "output", "Type_2")
    t3_out = os.path.join(proj_root, "wordbag", "output", "Type_3")

    WINDOW = 10
    MERGE_SHORT = True
    MIN_TOKENS = 3
    MIN_CHARS = 10

    print("[Paths]")
    print("  lex_json:", lex_json)
    print("  T1 in  :", t1_in, " -> ", t1_out)
    print("  T2 in  :", t2_in, " -> ", t2_out)
    print("  T3 in  :", t3_in, " -> ", t3_out)
    print("[Params]")
    print("  window:", WINDOW, "  merge_short:", MERGE_SHORT,
          "  min_tokens:", MIN_TOKENS, "  min_chars:", MIN_CHARS)

    process_type_dir(t1_in, t1_out, lex_json, WINDOW, MERGE_SHORT, MIN_TOKENS, MIN_CHARS)
    process_type_dir(t2_in, t2_out, lex_json, WINDOW, MERGE_SHORT, MIN_TOKENS, MIN_CHARS)
    process_type_dir(t3_in, t3_out, lex_json, WINDOW, MERGE_SHORT, MIN_TOKENS, MIN_CHARS)

if __name__ == "__main__":
    main()
