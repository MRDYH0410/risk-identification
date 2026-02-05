# -*- coding: utf-8 -*-
"""
MLM_light_risk_classifier.py

更新要点：
- risk_distance 逐句来自 wordbag（优先 sc_near_* / *dist* / *near* / prob/sim 翻转）
- sentences_scored_llm_ctx.csv 删除源 risk_bucket，仅保留 llm_pred_bucket 与四类规范概率
- 汇总输出为 a_mij__sentence_ctx.csv，比例列 a_mij，risk_bucket 统一四类规范名
- 新增并强制保留列 source_type（Type_1/Type_2/Type_3），两个输出表均含之
"""

import os, re, json, glob, random, difflib
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForMaskedLM,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
)
from torch.optim import AdamW

# ===================== 路径 =====================
BASE_DIR   = os.path.dirname(__file__)
PROJ_ROOT  = os.path.abspath(os.path.join(BASE_DIR, ".."))
WORDBAG_OUT_ROOT = os.path.join(PROJ_ROOT, "wordbag", "output")
RAW_TXT_ROOT     = os.path.join(PROJ_ROOT, "grasp_data", "output")
OUT_DIR    = os.path.join(PROJ_ROOT, "identification", "output")
CKPT_DIR   = os.path.join(PROJ_ROOT, "identification", "llm_ckpt")
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)

# ===================== 环境 =====================
OFFLINE = True
LOCAL_MODELS_DIR = os.path.join(os.path.expanduser("~"), "models")
LOCAL_MODEL_CANDIDATES = {
    "distilbert-base-uncased": [
        os.path.join(LOCAL_MODELS_DIR, "distilbert-local-mlm"),
        os.path.join(LOCAL_MODELS_DIR, "distilbert-local-clf"),
        os.path.join(LOCAL_MODELS_DIR, "distilbert-local"),
    ],
    "ProsusAI/finbert": [
        os.path.join(LOCAL_MODELS_DIR, "distilbert-local-mlm"),
        os.path.join(LOCAL_MODELS_DIR, "distilbert-local-clf"),
        os.path.join(LOCAL_MODELS_DIR, "distilbert-local"),
    ],
}

def set_offline_env() -> None:
    os.environ.setdefault("HF_HOME", os.path.join(PROJ_ROOT, ".hf_cache"))
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    if OFFLINE:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

def resolve_model_path(name_or_path: str, prefer_mlm: bool = False) -> str:
    ckpt_ctx = os.path.join(CKPT_DIR, "mlm_continued_ctx")
    ckpt_std = os.path.join(CKPT_DIR, "mlm_continued")
    for p in ([ckpt_ctx, ckpt_std] if prefer_mlm else [ckpt_std, ckpt_ctx]):
        if os.path.isdir(p): return p
    for p in LOCAL_MODEL_CANDIDATES.get(name_or_path, []):
        if os.path.isdir(p): return p
    return name_or_path

# ===================== 参数 =====================
DEVICE_PREFERENCE  = "cuda"
MODEL_NAME         = "distilbert-base-uncased"
CONTEXT_K          = 2
MAX_LENGTH         = 320
BATCH_SIZE         = 8
EPOCHS             = 3
LR                 = 2e-5
VAL_RATIO          = 0.2
SEED               = 42

DO_MLM_PRETRAIN    = True
MLM_MAX_TEXTS      = 250_000
MLM_EPOCHS         = 1
MLM_BATCH          = 12
MLM_LR             = 5e-5
MLM_MASK_PROB      = 0.15

MIN_COUNT          = 5
LIMIT_PER_CLASS    = 5000
FREEZE_ENCODER     = False
GRAD_ACCUM_STEPS   = 2
PRED_SHARD_SIZE    = 20000

# ===================== 工具 =====================
def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if DEVICE_PREFERENCE == "cuda" and torch.cuda.is_available() else "cpu")

def list_type_dirs(root: str) -> List[str]:
    ret = []
    for d in ["Type_1","Type_2","Type_3"]:
        p = os.path.join(root, d)
        if os.path.isdir(p): ret.append(p)
    return ret

def _infer_doc_id_from_filename(path: str) -> str:
    base = os.path.basename(path)
    base = re.sub(r"\.sentences\.csv$","", base)
    base = re.sub(r"\.csv$","", base)
    base = re.sub(r"\.txt$","", base)
    return base

def infer_company_id_from_doc(doc_id: str) -> str:
    base = doc_id.split("_")[0]
    if "," in base: base = base.split(",")[0]
    return re.sub(r"\s+"," ", base).strip()

def _read_csv_robust(path: str) -> Optional[pd.DataFrame]:
    try:
        if os.path.getsize(path) < 5: return None
    except OSError:
        return None
    encs = ["utf-8","utf-8-sig","latin1"]
    seps = [None, ",", "\t", "|"]
    for e in encs:
        for s in seps:
            try:
                df = pd.read_csv(path, encoding=e, sep=s, engine="python", on_bad_lines="skip")
                if df is not None and df.shape[1] > 0: return df
            except Exception:
                continue
    return None

# ========= 规范四类 =========
CANON_KEYS = ["risk_uncertainty","risk_legal","risk_constraint","risk_external"]

def canonicalize_label(name: str) -> Optional[str]:
    if not name: return None
    t = re.sub(r"[^a-z0-9]+", "", name.lower())
    if "q1" in t or "uncertain" in t:                      return "risk_uncertainty"
    if "q2" in t or "legal" in t:                           return "risk_legal"
    if "q3" in t or "constraint" in t or "operational" in t:return "risk_constraint"
    if "q4" in t or "external" in t or "macro" in t:        return "risk_external"
    return None

def _parse_source_type_from_path(path: str) -> str:
    m = re.search(r"(Type_[123])", path, re.IGNORECASE)
    return m.group(1) if m else ""

def _ensure_source_type(df: pd.DataFrame) -> pd.DataFrame:
    """强制填充/修正 source_type：优先现有列；否则从 source_csv_path / doc_id 反推。"""
    df = df.copy()
    if "source_type" not in df.columns or df["source_type"].isna().all() or (df["source_type"].astype(str).str.len()==0).all():
        if "source_csv_path" in df.columns:
            df["source_type"] = df["source_csv_path"].astype(str).apply(_parse_source_type_from_path)
        else:
            # 兜底：根据 doc_id 在所有候选目录里寻找匹配路径
            doc2type = {}
            for tdir in list_type_dirs(WORDBAG_OUT_ROOT):
                for p in glob.glob(os.path.join(tdir, "*.sentences.csv")) + glob.glob(os.path.join(tdir, "*.txt.sentences.csv")):
                    doc2type[_infer_doc_id_from_filename(p)] = _parse_source_type_from_path(p)
            df["source_type"] = df.get("doc_id","").astype(str).map(doc2type).fillna("")
    return df

# ---------- 统计公司数量（按 Type_1/2/3，对应 z=1/2/3） ----------
TYPE_TO_Z = {
    "Type_1": 1,
    "Type_2": 2,
    "Type_3": 3,
}

def debug_company_counts(tag: str, df: pd.DataFrame) -> None:
    """
    控制台输出:
      - 当前阶段一共有多少家公司
      - Type_1 / Type_2 / Type_3 各有多少家公司（对应 z=1/2/3）
    """
    if "company_id" not in df.columns or "source_type" not in df.columns:
        print(f"[DEBUG] {tag}: missing company_id or source_type; skip debug.")
        return

    df = df.copy()
    df["source_type"] = df["source_type"].astype(str)

    total_companies = df["company_id"].nunique()
    print(f"[DEBUG] {tag}: total companies = {total_companies}")

    for t in ["Type_1", "Type_2", "Type_3"]:
        mask = df["source_type"].str.upper() == t.upper()
        n_comp_t = df.loc[mask, "company_id"].nunique()
        z_val = TYPE_TO_Z.get(t)
        if z_val is not None:
            print(f"    {t} (z={z_val}): {n_comp_t} companies")
        else:
            print(f"    {t}: {n_comp_t} companies")

# ========= 从 WORDBAG 抽取逐句 risk_distance =========
def _pull_wordbag_distance(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    cols = list(df.columns)

    # 1) 直接 distance 列
    for c in cols:
        if re.fullmatch(r"(?i)(risk_)?distance|risk_dist|dist", c):
            dist = pd.to_numeric(df[c], errors="coerce").astype(float)
            return pd.Series(["unknown"]*len(df), index=df.index), dist.fillna(1.0)

    # 2) 组合 “near/dist” 数值列
    near_cols = [c for c in cols if re.search(r"(?i)(^sc_)?near|dist", c)]
    num_near = []
    for c in near_cols:
        try:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any(): num_near.append(c)
        except Exception:
            pass
    if num_near:
        sub = df[num_near].apply(pd.to_numeric, errors="coerce")
        dist = sub.min(axis=1).astype(float).fillna(1.0)
        min_idx = sub.values.argmin(axis=1)
        names = [num_near[i] if 0 <= i < len(num_near) else "" for i in min_idx]
        rb = [canonicalize_label(n) or "unknown" for n in names]
        return pd.Series(rb, index=df.index), dist

    # 3) 概率/相似度翻转
    sim_like = [c for c in cols if re.match(r"(?i)^(prob|sim|score)[_\-]?", c)]
    num_sim  = []
    for c in sim_like:
        try:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().any(): num_sim.append(c)
        except Exception:
            pass
    if num_sim:
        sub = df[num_sim].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        mx = sub.max(axis=1)
        dist = (1.0 - mx).clip(0, 1).astype(float)
        return pd.Series(["unknown"]*len(df), index=df.index), dist

    # 4) 兜底
    return pd.Series(["unknown"]*len(df), index=df.index), pd.Series([1.0]*len(df), index=df.index, dtype=float)

def load_all_sentence_csvs(wordbag_out_root: str) -> pd.DataFrame:
    skip_log = os.path.join(OUT_DIR, "loader_skip_log.txt")
    skips: List[str] = []
    paths: List[str] = []
    for tdir in list_type_dirs(wordbag_out_root):
        one = os.path.join(tdir, "sentences_scored.csv")
        if os.path.isfile(one): paths.append(one)
        paths.extend(sorted(glob.glob(os.path.join(tdir, "*.txt.sentences.csv"))))
    if not paths:
        raise FileNotFoundError(f"No sentence CSVs under {wordbag_out_root}")

    dfs = []
    for p in paths:
        df = _read_csv_robust(p)
        if df is None or df.empty or df.shape[1]==0:
            skips.append(f"[EMPTY] {p}"); continue

        # 标准化基本列
        if "text" not in df.columns:
            for c in ["sentence","sent_text","content","line","raw"]:
                if c in df.columns: df = df.rename(columns={c:"text"}); break
        if "text" not in df.columns:
            skips.append(f"[MISS text] {p}"); continue
        if "sent_idx" not in df.columns:
            for c in ["s_idx","sentence_idx","index","idx","line_no","lineno"]:
                if c in df.columns: df = df.rename(columns={c:"sent_idx"}); break
            if "sent_idx" not in df.columns: df["sent_idx"] = range(len(df))
        if "doc_id" not in df.columns:
            df["doc_id"] = _infer_doc_id_from_filename(p)

        # —— 关键：从 wordbag 提取距离 —— #
        rb, rd = _pull_wordbag_distance(df)
        df["risk_distance"] = rd.astype(float).fillna(1.0)
        if "risk_bucket" not in df.columns:
            df["risk_bucket"] = rb

        # 附加来源路径 & Type
        df["source_csv_path"] = p
        df["source_type"]     = _parse_source_type_from_path(p)

        df["text"] = df["text"].astype(str)
        df = df[df["text"].str.strip().str.len()>0].copy()
        df["sent_idx"] = pd.to_numeric(df["sent_idx"], errors="coerce").fillna(0).astype(int)
        df["risk_bucket"] = df["risk_bucket"].astype(str).str.strip().replace({"": "unknown"})
        df["company_id"] = df["doc_id"].apply(infer_company_id_from_doc)
        dfs.append(df)

    if skips:
        with open(skip_log,"w",encoding="utf-8") as f: f.write("\n".join(skips))
    if not dfs:
        raise RuntimeError("No valid CSV loaded. Check loader_skip_log.txt")
    df_all = pd.concat(dfs, ignore_index=True).sort_values(["doc_id","sent_idx"]).reset_index(drop=True)
    # —— 强制确保 source_type 存在 —— #
    df_all = _ensure_source_type(df_all)
    return df_all

# ===================== 说话人识别（略） =====================
ROLE_SET = ["CEO","CFO","COO","CTO","CIO","CMO","Chair","President","SVP","EVP","VP","Director",
            "Treasurer","Controller","General Counsel","Managing Director","Head","IR","Analyst","Operator"]
ROLE_PRIOR = {"Operator":0.10,"Analyst":0.35,"CEO":0.18,"CFO":0.18,"IR":0.14}
ROLE_ALIAS_MAP = {
    "chief executive officer":"CEO","ceo":"CEO",
    "chief financial officer":"CFO","cfo":"CFO",
    "chief operating officer":"COO","coo":"COO",
    "chief technology officer":"CTO","cto":"CTO",
    "chief information officer":"CIO","cio":"CIO",
    "chief marketing officer":"CMO","cmo":"CMO",
    "chairman":"Chair","chair":"Chair","president":"President",
    "senior vice president":"SVP","svp":"SVP","executive vice president":"EVP","evp":"EVP",
    "vice president":"VP","vp":"VP","general counsel":"General Counsel",
    "managing director":"Managing Director","head":"Head","investor relations":"IR","ir":"IR",
    "analyst":"Analyst","operator":"Operator","moderator":"Operator","host":"Operator",
    "主持人":"Operator","问答":"Analyst","分析师":"Analyst","投资者关系":"IR","董事长":"Chair","总裁":"President",
    "首席执行官":"CEO","首席财务官":"CFO","首席运营官":"COO","首席技术官":"CTO","首席信息官":"CIO","首席市场官":"CMO"
}
BROKER_CUES = ["Morgan Stanley","Goldman Sachs","J.P. Morgan","JPMorgan","Bank of America","BofA",
    "Barclays","UBS","Citi","Citigroup","Deutsche Bank","Wolfe","Jefferies","RBC","TD","BMO","Mizuho",
    "Nomura","Macquarie","Credit Suisse","Evercore","Piper Sandler","Raymond James","Cowen","Needham","Truist","Stephens"]
LEXICON = {
    "Operator":["welcome to","please stand by","we will begin","operator instructions","question-and-answer","q&a session","our first question","the conference call"],
    "Analyst":["question","questions","q:","follow-up","from ","on the line"] + BROKER_CUES,
    "CFO":["guidance","margin","gross margin","sg&a","capex","liquidity","cash flow","ebitda","interest","balance sheet","cost of","tax rate","debt","leverage","free cash","inventory","working capital"],
    "CEO":["strategy","strategic","long-term","vision","mission","customers","market opportunity","transformation","innovation","priorities","roadmap"],
    "IR":["investor relations","ir team","prepared remarks","safe harbor","forward-looking statements"],
}

def _canon_role(s: str) -> str:
    if not s: return ""
    t = re.sub(r"[^A-Za-z\s]"," ", s).lower().strip()
    t = re.sub(r"\s+"," ", t)
    for k,v in ROLE_ALIAS_MAP.items():
        if k in t: return v
    return ""

def _sanitize_role(s: str) -> str:
    r = _canon_role(s); return r if r in ROLE_SET else ""

def _infer_role_by_lexicon(text: str) -> Optional[str]:
    t = (text or "").lower()
    for role in ["Operator","Analyst"]:
        for kw in LEXICON[role]:
            if kw.lower() in t: return role
    for role in ["CFO","CEO","IR"]:
        if sum(1 for kw in LEXICON[role] if kw.lower() in t) >= 2: return role
    return None

def _score_roles(text: str, allowed: List[str]) -> Dict[str, float]:
    t = (text or "").lower(); scores = {r:0.0 for r in allowed}
    if "?" in t or t.strip().startswith(("q:", "question:")):
        if "Analyst" in scores: scores["Analyst"] += 2.5
    if any(p in t for p in ["welcome to","please stand by","operator instructions"]):
        if "Operator" in scores: scores["Operator"] += 3.0
    if len(re.findall(r"\d+(\.\d+)?", t)) >= 2 and "CFO" in scores: scores["CFO"] += 0.6
    for role in ["Operator","Analyst","CFO","CEO","IR"]:
        if role in scores: scores[role] += sum(1.0 for kw in LEXICON[role] if kw.lower() in t)
    return scores

def guess_raw_txt_path(doc_id: str) -> Optional[str]:
    for tdir in list_type_dirs(RAW_TXT_ROOT):
        for pat in [f"{doc_id}.txt", f"{doc_id}*.txt", f"{doc_id.split('_')[0]}*.txt"]:
            cand = glob.glob(os.path.join(tdir, pat))
            if cand: return cand[0]
    return None

def parse_participants_header(lines: List[str]) -> Dict[str, str]:
    name2role: Dict[str,str] = {}
    block = None
    header_pat = re.compile(r"(?i)^(participants|company participants|corporate participants|conference call participants|analysts|q&a participants)\s*$")
    for raw in lines:
        line = (raw or "").strip()
        if not line: continue
        if header_pat.match(line): block = line.lower(); continue
        if block and re.match(r"(?i)^(presentation|prepared remarks|operator|conference call|q&a|question)", line):
            block = None; continue
        if not block: continue
        s = re.sub(r"\s*\[\d+\]\s*$", "", line)
        parts = re.split(r"\s+[-—,]\s+", s, maxsplit=1)
        if len(parts) >= 2:
            name, right = parts[0].strip(), parts[1].strip()
            role = _sanitize_role(right)
            if not role and ("analyst" in right.lower() or any(b.lower() in right.lower() for b in BROKER_CUES)):
                role = "Analyst"
            if not role and "analyst" in block: role = "Analyst"
            if not role and "participants" in block: role = _sanitize_role(right) or _canon_role(right) or "IR"
            if name: name2role[name.lower()] = role if role in ROLE_SET else "IR"
    return name2role

def extract_switch_role(line: str) -> Tuple[Optional[str], Optional[str]]:
    s = (line or "").strip()
    if not s: return None, None
    if re.fullmatch(r"(?i)operator|moderator|host", s): return None, "Operator"
    if re.match(r"(?i)^(q:|question:|questions?:)$", s): return None, "Analyst"
    if re.match(r"^[A-Z][A-Za-z\.\-\'\s]+:\s*$", s): return s[:-1].strip(), None
    m = re.match(r"^([A-Z][A-Za-z\.\-\'\s]+)\s+[-—,]\s+(.+)$", s)
    if m:
        name, right = m.group(1).strip(), m.group(2).strip()
        role = _sanitize_role(right) or _canon_role(right)
        if not role and any(b.lower() in right.lower() for b in [x.lower() for x in BROKER_CUES]): role = "Analyst"
        return name, (role if role in ROLE_SET else None)
    return None, None

def naive_sentence_split(text: str) -> List[str]:
    ph = {"e.g.":"eg_ph","i.e.":"ie_ph","U.S.":"US_ph","U.K.":"UK_ph","Mr.":"Mr_ph","Ms.":"Ms_ph"}
    tmp = text
    for k,v in ph.items(): tmp = tmp.replace(k,v)
    parts = re.split(r"(?<=[。！？!?\.])\s+", tmp)
    out = []
    for p in parts:
        if not p: continue
        for k,v in ph.items(): p = p.replace(v,k)
        p = p.strip()
        if len(p) < 2: continue
        out.append(p)
    return out

def sentence_roles_from_raw(doc_path: str) -> Tuple[List[str], List[str]]:
    with open(doc_path,"r",encoding="utf-8",errors="ignore") as f:
        lines = f.read().splitlines()
    name2role = parse_participants_header(lines); known = list(name2role.keys())
    def map_name_to_role(name_raw: str) -> Optional[str]:
        if not name_raw: return None
        key = (name_raw or "").lower().strip()
        if key in name2role: return name2role[key]
        if known:
            mch = difflib.get_close_matches(key, known, n=1, cutoff=0.75)
            if mch: return name2role[mch[0]]
        if any(b.lower() in key for b in [x.lower() for x in BROKER_CUES]): return "Analyst"
        return None
    current_role = "IR"; buf, para_role_pairs = [], []
    def flush():
        nonlocal buf, current_role
        if not buf: return
        text = " ".join(buf).strip()
        if text: para_role_pairs.append((text, current_role))
        buf = []
    sep_pat = re.compile(r"^\s*[-=]{3,}\s*$")
    junk = [re.compile(p, re.I) for p in [
        r"^\s*questions? and answers?\s*$", r"^\s*\(operator instructions\)\s*$",
        r"^\s*forward[- ]looking statements.*$", r"^\s*copyright.*$",
        r"^\s*participants\s*$", r"^\s*(presentation|prepared remarks)\s*$",
    ]]
    for raw in lines:
        line = (raw or "").strip()
        if not line: continue
        if sep_pat.match(line): flush(); continue
        if any(r.match(line) for r in junk): continue
        name, role = extract_switch_role(line)
        if name is not None or role is not None:
            flush()
            if name is not None:
                r = map_name_to_role(name) or role
                if r in ROLE_SET: current_role = r
            elif role in ROLE_SET: current_role = role
            continue
        if re.match(r"(?i)^(q:|question:)\s*", line): flush(); current_role = "Analyst"; continue
        if re.match(r"(?i)^operator:\s*", line): flush(); current_role = "Operator"; continue
        buf.append(line)
        if re.search(r"[。\.!?]$", line) and sum(len(x) for x in buf) > 200: flush()
    flush()
    roles, texts = [], []
    for para, role in para_role_pairs:
        sents = naive_sentence_split(para)
        for s in sents:
            roles.append(role if role in ROLE_SET else "IR")
            texts.append(s)
    # 传播与兜底
    W = 3
    for i, r in enumerate(roles):
        if r in ROLE_SET: continue
        for d in range(1, W+1):
            j = i-d; k = i+d
            if j >= 0 and roles[j] in ROLE_SET: roles[i] = roles[j]; break
            if k < len(roles) and roles[k] in ROLE_SET: roles[i] = roles[k]; break
        if roles[i] in ROLE_SET: continue
        guess = _infer_role_by_lexicon(texts[i]); roles[i] = guess if (guess in ROLE_SET) else "IR"
    return roles, texts

def _rebalance_roles(texts: List[str], init_roles: List[str], allowed_roles: List[str]) -> List[str]:
    n = len(texts); roles = init_roles[:]
    allowed = [r for r in allowed_roles if r in ROLE_SET] or ["Operator","Analyst","CEO","CFO","IR"]
    cnt = {r:0 for r in allowed}
    for r in roles:
        if r in cnt: cnt[r] += 1
    target = {r:int(round(ROLE_PRIOR.get(r, 0.05) * n)) for r in allowed}
    for r in allowed:
        target[r] = max(1, target.get(r, 0))
    delta = n - sum(target.values())
    order = [r for r in ["Analyst","Operator","CEO","CFO","IR"] if r in allowed]
    i = 0
    while delta != 0 and order:
        r = order[i % len(order)]
        if delta > 0: target[r] += 1; delta -= 1
        else:
            if target[r] > 1: target[r] -= 1; delta += 1
        i += 1
        if i > 10000: break
    deficits = {r:max(0, target[r]-cnt.get(r,0)) for r in allowed}
    need_roles = []
    for r, k in deficits.items(): need_roles += [r]*k
    if not need_roles: return roles
    candidate_idx = [i for i,r in enumerate(roles) if r not in allowed or r=="IR"]
    scored = [(i, _score_roles(texts[i], allowed)) for i in candidate_idx]
    for target_role in need_roles:
        best_i, best_s, best_pos = None, -1e9, -1
        for pos, (idx, sc) in enumerate(scored):
            if idx is None: continue
            s = sc.get(target_role, 0.0)
            if s > best_s: best_s, best_i, best_pos = s, idx, pos
        if best_i is not None:
            roles[best_i] = target_role; scored[best_pos] = (None, {})
        else:
            for pos, (idx, _) in enumerate(scored):
                if idx is not None: roles[idx] = target_role; scored[pos] = (None, {}); break
    return roles

def _validate_and_fix_roles(roles: List[str], allowed_roles: List[str]) -> List[str]:
    allowed = [r for r in allowed_roles if r in ROLE_SET] or ["Operator","Analyst","CEO","CFO","IR"]
    roles = [r if r in ROLE_SET else "IR" for r in roles]
    if len(set(roles)) == 1 and len(roles) >= 10:
        for i in range(len(roles)):
            if i % 7 == 0: roles[i] = allowed[i % len(allowed)]
    return roles

def attach_speaker_roles(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for doc_id, g in df.groupby("doc_id"):
        rawp = guess_raw_txt_path(doc_id)
        roles = []; allowed_roles = []
        texts_sorted = g.sort_values(["sent_idx"])["text"].astype(str).tolist()
        if rawp and os.path.isfile(rawp):
            try:
                roles, _ = sentence_roles_from_raw(rawp)
                with open(rawp,"r",encoding="utf-8",errors="ignore") as f:
                    lines = f.read().splitlines()
                name2role = parse_participants_header(lines)
                allowed_roles = sorted({r for r in name2role.values() if r in ROLE_SET})
            except Exception:
                roles, allowed_roles = [], []
        n = len(g)
        if not roles: roles = ["IR"] * n
        if len(roles) < n: roles += ["IR"] * (n-len(roles))
        elif len(roles) > n: roles = roles[:n]
        roles = [r if r in ROLE_SET else "IR" for r in roles]
        roles = _rebalance_roles(texts_sorted, roles, allowed_roles)
        roles = _validate_and_fix_roles(roles, allowed_roles)
        gg = g.sort_values(["sent_idx"]).copy()
        gg["speaker_role"] = roles[:len(gg)]
        out.append(gg)
    return pd.concat(out, ignore_index=True)

# ===================== 上下文与数据集 =====================
def build_context_inputs(df: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    df = df.sort_values(["doc_id","sent_idx"]).reset_index(drop=True)
    rows = []
    for _, g in df.groupby("doc_id"):
        g = g.reset_index(drop=True)
        texts = g["text"].astype(str).tolist()
        roles  = g["speaker_role"].fillna("IR").tolist()
        for i in range(len(g)):
            left  = texts[max(0, i-k): i]
            right = texts[i+1: i+1+k]
            cur   = texts[i]; role  = roles[i]
            inp = f"[SPEAKER={role}] "
            if left:  inp += " ".join(left) + " [CTX_BEFORE] "
            inp += cur
            if right: inp += " [CTX_AFTER] " + " ".join(right)
            row = g.loc[i].to_dict()
            row["input_with_ctx"] = inp
            rows.append(row)
    out = pd.DataFrame(rows)
    # —— 强制保留 source_type —— #
    out = _ensure_source_type(out)
    return out

class RiskDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label2id: Dict[str,int], tokenizer, max_length: int = 320):
        self.df = df.reset_index(drop=True); self.label2id = label2id
        self.tok = tokenizer; self.max_length = max_length
        self.labels = [label2id[y] for y in self.df["risk_bucket"].tolist()]
        d = self.df["risk_distance"].astype(float).fillna(1.0).values
        conf = np.clip(1.0 - d, 0.05, 1.0)
        self.sample_weights = conf.astype(np.float32)
    def __len__(self): return len(self.df)
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = str(self.df.at[idx, "input_with_ctx"])
        enc = self.tok(text, padding=False, truncation=True, max_length=self.max_length, return_tensors="pt")
        item = {k: v.squeeze(0) for k,v in enc.items()}
        item["labels"]  = torch.tensor(self.labels[idx], dtype=torch.long)
        item["weights"] = torch.tensor(self.sample_weights[idx], dtype=torch.float32)
        return item

class WeightedCELoss(nn.Module):
    def __init__(self): super().__init__(); self.ce = nn.CrossEntropyLoss(reduction="none")
    def forward(self, logits, labels, weights):
        loss_vec = self.ce(logits, labels)
        return (loss_vec * weights).mean()

def stratified_cap(df: pd.DataFrame, limit_per_class: Optional[int]) -> pd.DataFrame:
    if not limit_per_class: return df
    xs = []
    for y, g in df.groupby("risk_bucket"):
        xs.append(g.sample(n=limit_per_class, random_state=SEED) if len(g) > limit_per_class else g)
    return pd.concat(xs, ignore_index=True)

def split_train_val(df: pd.DataFrame, val_ratio: float = 0.2, seed: int = 42):
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_parts, train_parts = [], []
    for _, g in df.groupby("risk_bucket"):
        n = len(g); m = max(1, int(n * val_ratio))
        val_parts.append(g.iloc[:m]); train_parts.append(g.iloc[m:])
    val_df = pd.concat(val_parts).reset_index(drop=True)
    train_df = pd.concat(train_parts).reset_index(drop=True)
    val_df = val_df.sample(frac=1.0, random_state=seed+1).reset_index(drop=True)
    train_df = train_df.sample(frac=1.0, random_state=seed+2).reset_index(drop=True)
    return train_df, val_df

def freeze_encoder_layers(model):
    for n,p in model.named_parameters():
        if "classifier" in n or "score" in n: p.requires_grad = True
        else: p.requires_grad = False

# ===================== MLM 继续预训练 =====================
class MLMDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 320):
        self.texts = texts; self.tok = tokenizer; self.max_length = max_length
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        txt = self.texts[idx]
        enc = self.tok(txt, padding=False, truncation=True, max_length=self.max_length, return_tensors="pt")
        return {k: v.squeeze(0) for k, v in enc.items()}

def mlm_collate(batch, tokenizer, mask_prob: float, device: torch.device):
    input_ids = [x["input_ids"] for x in batch]
    attn_masks = [x["attention_mask"] for x in batch]
    maxlen = max(t.size(0) for t in input_ids)
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    def pad_to(t, L, padv):
        if t.size(0) == L: return t
        return torch.cat([t, torch.full((L - t.size(0),), padv, dtype=t.dtype)], dim=0)
    input_ids = torch.stack([pad_to(t, maxlen, pad_id) for t in input_ids], dim=0)
    attn_masks = torch.stack([pad_to(t, maxlen, 0) for t in attn_masks], dim=0)
    labels = input_ids.clone()
    prob_mat = torch.full(labels.shape, mask_prob)
    special = torch.tensor([
        tokenizer.get_special_tokens_mask(val.tolist(), already_has_special_tokens=True)
        for val in labels
    ], dtype=torch.bool)
    prob_mat.masked_fill_(special, 0.0)
    masked = torch.bernoulli(prob_mat).bool()
    labels[~masked] = -100
    ids_masked = input_ids.clone()
    mask_token_id = tokenizer.mask_token_id or pad_id
    ids_masked[masked] = mask_token_id
    return {"input_ids": ids_masked.to(device), "attention_mask": attn_masks.to(device), "labels": labels.to(device)}

def run_mlm_pretrain(base_model: str, texts: List[str], save_dir: str,
                     epochs=1, bs=16, lr=5e-5, mask_prob=0.15, device=None):
    base = resolve_model_path(base_model, prefer_mlm=False)
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True, local_files_only=OFFLINE)
    special_tokens = ["[CTX_BEFORE]", "[CTX_AFTER]"] + [f"[SPEAKER={r}]" for r in
        ["CEO","CFO","IR","Analyst","Operator","Other","Unknown","President","Chair","Director","SVP","EVP","VP"]]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
    model = AutoModelForMaskedLM.from_pretrained(base, local_files_only=OFFLINE)
    model.resize_token_embeddings(len(tokenizer))
    device = device or get_device()
    model.to(device)
    ds = MLMDataset(texts, tokenizer, max_length=MAX_LENGTH)
    collate_fn = lambda batch: mlm_collate(batch, tokenizer, mask_prob, device)
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=0, collate_fn=collate_fn)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps  = len(dl) * max(1, epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1*total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))
    model.train()
    for ep in range(1, epochs+1):
        total_loss = 0.0
        for step, batch in enumerate(dl, start=1):
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                out = model(**batch); loss = out.loss
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            scheduler.step()
            total_loss += float(loss.item())
            if step % 200 == 0: print(f"[MLM ep {ep} step {step}] loss={total_loss/step:.4f}")
        print(f"[MLM] epoch {ep} avg_loss={total_loss/max(1,len(dl)):.4f}")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir); tokenizer.save_pretrained(save_dir)
    print(f"[MLM] saved checkpoint -> {save_dir}")
    return save_dir

# ===================== 训练 & 评估 =====================
def ensure_labels(df_ctx: pd.DataFrame) -> pd.DataFrame:
    df_ctx = df_ctx.copy()
    if "risk_bucket" in df_ctx.columns:
        uniq = df_ctx["risk_bucket"].astype(str).str.lower().unique().tolist()
        if not (len(uniq) == 1 and uniq[0] in {"unknown","nan",""}): return df_ctx
    d = pd.to_numeric(df_ctx.get("risk_distance"), errors="coerce").fillna(1.0).astype(float)
    rng = np.random.default_rng(SEED); d_jitter = d.values + rng.normal(0.0, 1e-8, size=len(d))
    try:
        lab = ["risk_q1","risk_q2","risk_q3","risk_q4"]
        df_ctx["risk_bucket"] = pd.qcut(d_jitter, q=4, labels=lab, duplicates="drop").astype(str)
        if df_ctx["risk_bucket"].nunique() >= 2: return df_ctx
    except Exception: pass
    lens = df_ctx["text"].astype(str).str.len().to_numpy()
    lens_jitter = lens + rng.normal(0.0, 1e-8, size=len(lens))
    try:
        lab = ["len_q1","len_q2","len_q3","len_q4"]
        df_ctx["risk_bucket"] = pd.qcut(lens_jitter, q=4, labels=lab, duplicates="drop").astype(str)
        if df_ctx["risk_bucket"].nunique() >= 2: return df_ctx
    except Exception: pass
    mid = np.median(lens_jitter)
    df_ctx["risk_bucket"] = np.where(lens_jitter <= mid, "bin_a", "bin_b")
    return df_ctx

def evaluate(model, dataloader, device):
    model.eval(); all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloader:
            ids  = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y    = batch["labels"].to(device)
            logits = model(input_ids=ids, attention_mask=attn).logits
            preds  = logits.argmax(dim=-1)
            all_preds.append(preds.cpu()); all_labels.append(y.cpu())
    y_pred = torch.cat(all_preds).numpy(); y_true = torch.cat(all_labels).numpy()
    acc = float((y_pred == y_true).mean())
    from sklearn.metrics import f1_score
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    return {"accuracy": acc, "macro_f1": f1}

def fit_model(df_ctx: pd.DataFrame, init_from: str, device):
    df_ctx = ensure_labels(df_ctx)
    vc = df_ctx["risk_bucket"].value_counts()
    kept = vc[vc >= MIN_COUNT].index.tolist()
    df_ctx = df_ctx[df_ctx["risk_bucket"].isin(kept)].reset_index(drop=True)
    df_ctx = stratified_cap(df_ctx, LIMIT_PER_CLASS)
    print("Label counts after filtering:\n", df_ctx["risk_bucket"].value_counts())

    labels   = sorted(df_ctx["risk_bucket"].unique().tolist())
    label2id = {y:i for i,y in enumerate(labels)}
    id2label = {i:y for y,i in label2id.items()}

    base = resolve_model_path(init_from, prefer_mlm=True)
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True, local_files_only=OFFLINE)
    special_tokens = ["[CTX_BEFORE]", "[CTX_AFTER]"] + [f"[SPEAKER={r}]" for r in
        ["CEO","CFO","IR","Analyst","Operator","Other","Unknown","President","Chair","Director","SVP","EVP","VP"]]
    tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})

    model = AutoModelForSequenceClassification.from_pretrained(
        base, num_labels=len(labels), id2label=id2label, label2id=label2id,
        problem_type="single_label_classification", ignore_mismatched_sizes=True, local_files_only=OFFLINE
    )
    model.resize_token_embeddings(len(tokenizer))
    if FREEZE_ENCODER: freeze_encoder_layers(model)

    train_df, val_df = split_train_val(df_ctx, val_ratio=VAL_RATIO, seed=SEED)
    train_ds = RiskDataset(train_df, label2id, tokenizer, max_length=MAX_LENGTH)
    val_ds   = RiskDataset(val_df,   label2id, tokenizer, max_length=MAX_LENGTH)
    collator = DataCollatorWithPadding(tokenizer, padding="longest",
                                       pad_to_multiple_of=8 if device.type=="cuda" else None)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, collate_fn=collator)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collator)

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    total_steps  = max(1, len(train_loader)) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1*total_steps), total_steps)
    criterion = WeightedCELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    model.to(device)
    best_f1 = -1.0
    best_dir = os.path.join(CKPT_DIR, "best_model_sentence_ctx"); os.makedirs(best_dir, exist_ok=True)

    step_acc = 0
    for ep in range(1, EPOCHS+1):
        model.train(); total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(train_loader, start=1):
            ids  = batch["input_ids"].to(device)
            attn = batch["attention_mask"].to(device)
            y    = batch["labels"].to(device)
            w    = batch["weights"].to(device)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(input_ids=ids, attention_mask=attn).logits
                loss   = criterion(logits, y, w) / max(1, GRAD_ACCUM_STEPS)
            scaler.scale(loss).backward()
            step_acc += 1
            if step_acc % GRAD_ACCUM_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update()
                optimizer.zero_grad(); scheduler.step()
            total_loss += float(loss.item()) * max(1, GRAD_ACCUM_STEPS)
            if step % 100 == 0:
                print(f"[train ep {ep} step {step}] loss={total_loss/step:.4f}")
        metrics = evaluate(model, val_loader, device)
        print(f"[val ep {ep}] acc={metrics['accuracy']:.4f}  macroF1={metrics['macro_f1']:.4f}")
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            model.save_pretrained(best_dir); tokenizer.save_pretrained(best_dir)
            with open(os.path.join(best_dir, "label_map.json"), "w", encoding="utf-8") as f:
                json.dump({"label2id": label2id, "id2label": id2label}, f, ensure_ascii=False, indent=2)
            print(f"[SAVE] best checkpoint -> {best_dir} (macroF1={best_f1:.4f})")
    return tokenizer, model, label2id, id2label, best_dir

# ===================== 推理（四类聚合 & 删源桶） =====================
@torch.no_grad()
def predict_sharded_and_write(df_raw: pd.DataFrame, df_ctx: pd.DataFrame,
                              tokenizer, model, id2label: Dict[int,str],
                              out_csv_path: str, device, shard_size: int = 20000) -> pd.DataFrame:
    model.eval(); N = len(df_ctx)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    out_parts = []; bs = max(2, BATCH_SIZE)

    # —— 防御：确保 df_raw 有 source_type —— #
    df_raw = _ensure_source_type(df_raw)

    for start in range(0, N, shard_size):
        end = min(N, start + shard_size)
        chunk = df_ctx.iloc[start:end].copy()
        texts = chunk["input_with_ctx"].astype(str).tolist()

        probs_list = []
        for i in range(0, len(texts), bs):
            enc = tokenizer(texts[i:i+bs], padding=True, truncation=True,
                            max_length=MAX_LENGTH, return_tensors="pt")
            ids  = enc["input_ids"].to(device)
            attn = enc["attention_mask"].to(device)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(input_ids=ids, attention_mask=attn).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()
            probs_list.append(probs)

        all_probs = np.vstack(probs_list)
        label_names = [id2label[i] for i in range(all_probs.shape[1])]
        canon_map = [canonicalize_label(n) for n in label_names]

        canon_probs = np.zeros((all_probs.shape[0], 4), dtype=np.float32)
        idx_map = {k:i for i,k in enumerate(CANON_KEYS)}
        for col_idx, canon_name in enumerate(canon_map):
            if canon_name is None: canon_name = "risk_uncertainty"
            canon_probs[:, idx_map[canon_name]] += all_probs[:, col_idx]

        prob_cols = [f"prob_{k}" for k in CANON_KEYS]
        df_probs = pd.DataFrame(canon_probs, columns=prob_cols, index=chunk.index)

        pred_id = canon_probs.argmax(axis=1)
        pred_lab = [CANON_KEYS[i] for i in pred_id]
        top3 = []
        for row in canon_probs:
            idx = np.argsort(-row)[:3]
            items = [f"{CANON_KEYS[j]}:{row[j]:.3f}" for j in idx]
            top3.append("; ".join(items))

        df_out = df_raw.iloc[start:end].copy()
        # 强制保留/填充 source_type
        df_out = _ensure_source_type(df_out)
        # 删除源 risk_bucket
        df_out = df_out.drop(columns=["risk_bucket"], errors="ignore")
        # 预测结果
        df_out["llm_pred_bucket"] = pred_lab
        df_out["llm_pred_prob"]   = canon_probs.max(axis=1)
        df_out["llm_top3"]        = top3
        df_out = pd.concat([df_out, df_probs], axis=1)

        # 可选：调整列顺序，把 source_type 放在 doc_id 右侧更显眼
        cols = list(df_out.columns)
        for k in ["source_type"]:
            if k in cols:
                cols.remove(k)
        if "doc_id" in df_out.columns:
            pos = cols.index("doc_id") + 1
            cols = cols[:pos] + ["source_type"] + cols[pos:]
        else:
            cols = ["source_type"] + cols
        df_out = df_out[cols]

        out_parts.append(df_out)
        mode = "w" if start == 0 else "a"; header = (start == 0)
        df_out.to_csv(out_csv_path, index=False, mode=mode, header=header)
        print(f"[WRITE shard] {out_csv_path} rows+={len(df_out)} (total so far {end})")
    return pd.concat(out_parts, ignore_index=True)

# ===================== 汇总 a_mij =====================
def compute_a_mij(prob_df: pd.DataFrame) -> pd.DataFrame:
    prob_df = _ensure_source_type(prob_df)
    prob_cols = [f"prob_{k}" for k in CANON_KEYS]
    rows = []
    for (m, role, stype), g in prob_df.groupby(["company_id","speaker_role","source_type"]):
        N = len(g)
        if N == 0: continue
        sums = g[prob_cols].sum(axis=0)
        total = float(sums.sum())
        props = (sums / total) if total > 0 else sums*0.0
        for k in CANON_KEYS:
            pc = f"prob_{k}"
            rows.append({
                "company_id": m,
                "role": role,
                "risk_bucket": k,
                "sum_prob": float(sums[pc]),
                "N_sent": int(N),
                "a_mij": float(props[pc]),
                "source_type": stype,
            })
    return pd.DataFrame(rows)

# ===================== 预览 =====================
def write_preview_csv(full_path: str, preview_path: str, n: int = 50) -> None:
    try:
        head_chunks = []; rows_needed = n
        for chunk in pd.read_csv(full_path, chunksize=5000):
            take = min(rows_needed, len(chunk))
            head_chunks.append(chunk.iloc[:take])
            rows_needed -= take
            if rows_needed <= 0: break
        if head_chunks:
            head_df = pd.concat(head_chunks, ignore_index=True)
            head_df.to_csv(preview_path, index=False)
            print(f"[PREVIEW] wrote {preview_path} ({len(head_df)} rows)")
        else:
            print(f"[PREVIEW] {full_path} seems empty; skip.")
    except Exception as e:
        print(f"[PREVIEW ERROR] {full_path}: {e}")

# ===================== 主流程 =====================
def main():
    print("[Paths]")
    print("  PROJ_ROOT        :", PROJ_ROOT)
    print("  WORDBAG_OUT_ROOT :", WORDBAG_OUT_ROOT)
    print("  RAW_TXT_ROOT     :", RAW_TXT_ROOT)
    print("  CKPT_DIR         :", CKPT_DIR)
    print("  OUT_DIR          :", OUT_DIR)
    print("  MODEL_NAME       :", MODEL_NAME)

    set_offline_env(); set_seed(SEED)
    device = get_device(); print("  DEVICE           :", device)

    # 1) 读取 wordbag 并强化说话人
    df_raw = load_all_sentence_csvs(WORDBAG_OUT_ROOT).sort_values(["doc_id","sent_idx"]).reset_index(drop=True)
    df_raw = attach_speaker_roles(df_raw)
    df_raw = _ensure_source_type(df_raw)

    # === 自测：输入公司数量（按 Type_1/2/3, 即 z=1/2/3） ===
    debug_company_counts("INPUT df_raw", df_raw)

    # 2) 上下文增强
    df_ctx = build_context_inputs(df_raw, k=CONTEXT_K)

    # 3) MLM 继续预训练（可选）
    init_model_path = MODEL_NAME
    mlm_ckpt_dir = os.path.join(CKPT_DIR, "mlm_continued_ctx")
    if DO_MLM_PRETRAIN:
        if os.path.isdir(mlm_ckpt_dir):
            print(f"[MLM] found existing checkpoint -> {mlm_ckpt_dir}; skip MLM pretraining.")
            init_model_path = mlm_ckpt_dir
        else:
            mlm_texts = df_ctx["input_with_ctx"].astype(str).tolist()
            if MLM_MAX_TEXTS and len(mlm_texts) > MLM_MAX_TEXTS:
                mlm_texts = random.sample(mlm_texts, MLM_MAX_TEXTS)
            init_model_path = run_mlm_pretrain(
                base_model=MODEL_NAME, texts=mlm_texts, save_dir=mlm_ckpt_dir,
                epochs=MLM_EPOCHS, bs=MLM_BATCH, lr=MLM_LR, mask_prob=MLM_MASK_PROB, device=device
            )
    else:
        init_model_path = resolve_model_path(MODEL_NAME, prefer_mlm=True)

    # 4) 监督微调
    tokenizer, model, label2id, id2label, ckpt_dir = fit_model(df_ctx, init_from=init_model_path, device=device)

    # 5) 推理写盘（删源 risk_bucket，输出四类概率 + source_type）
    out_sent_csv = os.path.join(OUT_DIR, "sentences_scored_llm_ctx.csv")
    prob_df = predict_sharded_and_write(df_raw, df_ctx, tokenizer, model, id2label,
                                        out_sent_csv, device=device, shard_size=PRED_SHARD_SIZE)

    # 5.1 预览
    preview_sent = os.path.join(OUT_DIR, "preview_sentences_scored_llm_ctx_head50.csv")
    write_preview_csv(out_sent_csv, preview_sent, n=50)

    # 6) 汇总 a_mij（含 source_type）
    a_df = compute_a_mij(prob_df)

    # === 自测：输出公司数量（按 Type_1/2/3, 即 z=1/2/3） ===
    debug_company_counts("OUTPUT a_mij", a_df)

    a_path = os.path.join(OUT_DIR, "a_mij__sentence_ctx.csv")
    a_df.to_csv(a_path, index=False)
    print(f"[WRITE] {a_path} rows={len(a_df)}")

    # 6.1 预览
    preview_a = os.path.join(OUT_DIR, "preview_a_mij__sentence_ctx_head50.csv")
    write_preview_csv(a_path, preview_a, n=50)

    print(f"[DONE] ✓ pipeline finished. ckpt -> {ckpt_dir}")
    print(f"Previews:\n  {preview_sent}\n  {preview_a}")

if __name__ == "__main__":
    main()
