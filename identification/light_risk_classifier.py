# -*- coding: utf-8 -*-
"""
轻量级（MLM-only）版本

目标：
- 只执行前三步：
  1) 读取句级输入 + 强化 speaker 标注
  2) 构造“上下文+角色提示”的输入
  3) 可选 MLM 继续预训练并落盘到:
     project/identification/llm_ckpt/mlm_continued_ctx

本文件会在拿到 mlm_continued_ctx 之后结束。
已删除 4/5/6 的全部逻辑与相关函数。

注意：
- 本版已加入“离线与本地模型配置”：默认仅用你本地的 distilbert 三件套或你刚训练出的 ckpt。
"""

import os, re, json, glob, random, difflib
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW


# =============== 基本路径 ===============
BASE_DIR   = os.path.dirname(__file__)
PROJ_ROOT  = os.path.abspath(os.path.join(BASE_DIR, ".."))
WORDBAG_OUT_ROOT = os.path.join(PROJ_ROOT, "wordbag", "output")
RAW_TXT_ROOT     = os.path.join(PROJ_ROOT, "grasp_data", "output")
OUT_DIR    = os.path.join(PROJ_ROOT, "identification", "output")
CKPT_DIR   = os.path.join(PROJ_ROOT, "identification", "llm_ckpt")
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

# =============== 离线与本地模型配置 ===============
OFFLINE = True                          # 服务器不能联网时设 True
HF_MIRROR = os.environ.get("HF_ENDPOINT", "")
LOCAL_MODELS_DIR = os.path.join(os.path.expanduser("~"), "models")

# 仅配置你现有的三个目录，设置优先级：mlm > clf > base
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
    """根据 OFFLINE 切换 HuggingFace 离线/镜像环境变量"""
    os.environ.setdefault("HF_HOME", os.path.join(PROJ_ROOT, ".hf_cache"))
    os.makedirs(os.environ["HF_HOME"], exist_ok=True)
    if OFFLINE:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
    else:
        if HF_MIRROR:
            os.environ["HF_ENDPOINT"] = HF_MIRROR

def resolve_model_path(name_or_path: str, prefer_mlm: bool = False) -> str:
    """
    返回最合适的本地/ckpt 路径；若都不存在则回传原字符串（仅当 OFFLINE=False 才可能去网）。
    优先级：
      1) 你训练输出的 ckpt：llm_ckpt/mlm_continued_ctx 或 mlm_continued
      2) /home/ubuntu/models 下的本地目录（优先 *-mlm）
      3) 原始名（仅 OFFLINE=False 有意义）
    """
    ckpt1 = os.path.join(CKPT_DIR, "mlm_continued_ctx")
    ckpt2 = os.path.join(CKPT_DIR, "mlm_continued")
    ckpts = [ckpt1, ckpt2] if prefer_mlm else [ckpt2, ckpt1]
    for p in ckpts:
        if os.path.isdir(p):
            return p

    cands = LOCAL_MODEL_CANDIDATES.get(name_or_path, [])
    for p in cands:
        if os.path.isdir(p):
            return p
    return name_or_path


# =============== 运行参数 ===============
DEVICE_PREFERENCE  = "cuda"  # "cuda" | "cpu"
MODEL_NAME         = "distilbert-base-uncased"
CONTEXT_K          = 2
MAX_LENGTH         = 320
SEED               = 42

# =============== MLM 继续预训练 ===============
DO_MLM_PRETRAIN    = True
MLM_MAX_TEXTS      = 250_000
MLM_EPOCHS         = 1
MLM_BATCH          = 12
MLM_LR             = 5e-5
MLM_MASK_PROB      = 0.15


# ----------------- 工具 -----------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_device():
    return torch.device("cuda" if DEVICE_PREFERENCE=="cuda" and torch.cuda.is_available() else "cpu")

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
    try:
        with open(path,"rb") as f:
            if not f.read(256).strip(): return None
    except Exception:
        pass
    encs = ["utf-8","utf-8-sig","latin1"]
    seps = [None, ",", "\t", "|"]
    for e in encs:
        for s in seps:
            try:
                df = pd.read_csv(path, encoding=e, sep=s, engine="python", on_bad_lines="skip")
                if df is not None and df.shape[1] > 0: return df
            except pd.errors.EmptyDataError:
                return None
            except Exception:
                continue
    return None

def _derive_risk_from_variant_cols(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    cols = list(df.columns)
    dist_cols = [c for c in cols if re.match(r"(?i)^dist[_\- ]", c)]
    sim_cols  = [c for c in cols if re.match(r"(?i)^(sim|score)[_\- ]", c)]
    if dist_cols:
        sub = df[dist_cols].apply(pd.to_numeric, errors="coerce").fillna(1.0)
        best_idx = sub.values.argmin(axis=1)
        bnames = [re.sub(r"(?i)^dist[_\- ]","",c) for c in sub.columns]
        rb = pd.Series([bnames[i] for i in best_idx], index=df.index)
        rd = sub.to_numpy()[range(len(df)), best_idx]
        return rb.astype(str), pd.Series(rd, index=df.index).astype(float)
    if sim_cols:
        sub = df[sim_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        best_idx = sub.values.argmax(axis=1)
        bnames = [re.sub(r"(?i)^(sim|score)[_\- ]","",c) for c in sub.columns]
        rb = pd.Series([bnames[i] for i in best_idx], index=df.index)
        ts = sub.to_numpy()[range(len(df)), best_idx]
        rd = 1.0 - ts
        return rb.astype(str), pd.Series(rd, index=df.index).astype(float)
    base = pd.Series(["unknown"] * len(df), index=df.index)
    dist = pd.to_numeric(df.get("risk_distance", pd.Series([1.0]*len(df), index=df.index)), errors="coerce").fillna(1.0)
    return base, dist

def load_all_sentence_csvs(wordbag_out_root: str) -> pd.DataFrame:
    """
    读取 Type_1/2/3 下的句级 CSV
    自动兼容不同列名并派生:
      - text
      - sent_idx
      - doc_id
      - risk_bucket
      - risk_distance
      - company_id
    """
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
            skips.append(f"[EMPTY] {p}");
            continue

        if "text" not in df.columns:
            for c in ["sentence","sent_text","content","line","raw"]:
                if c in df.columns:
                    df = df.rename(columns={c:"text"}); break

        if "text" not in df.columns:
            skips.append(f"[MISS text] {p}")
            continue

        if "sent_idx" not in df.columns:
            for c in ["s_idx","sentence_idx","index","idx","line_no","lineno"]:
                if c in df.columns:
                    df = df.rename(columns={c:"sent_idx"}); break
            if "sent_idx" not in df.columns:
                df["sent_idx"] = range(len(df))

        if "doc_id" not in df.columns:
            df["doc_id"] = _infer_doc_id_from_filename(p)

        if "risk_bucket" not in df.columns or "risk_distance" not in df.columns:
            rb, rd = _derive_risk_from_variant_cols(df)
            if "risk_bucket" not in df.columns: df["risk_bucket"] = rb
            if "risk_distance" not in df.columns: df["risk_distance"] = rd

        df["source_csv_path"] = p
        df["text"] = df["text"].astype(str)
        df = df[df["text"].str.strip().str.len()>0].copy()
        df["sent_idx"] = pd.to_numeric(df["sent_idx"], errors="coerce").fillna(0).astype(int)
        df["risk_distance"] = pd.to_numeric(df["risk_distance"], errors="coerce").fillna(1.0).astype(float)
        df["risk_bucket"] = df["risk_bucket"].astype(str).str.strip().replace({"": "unknown"})
        df["company_id"] = df["doc_id"].apply(infer_company_id_from_doc)
        dfs.append(df)

    if skips:
        with open(skip_log,"w",encoding="utf-8") as f:
            f.write("\n".join(skips))

    if not dfs:
        raise RuntimeError("No valid CSV loaded. Check loader_skip_log.txt")

    df_all = pd.concat(dfs, ignore_index=True).sort_values(["doc_id","sent_idx"]).reset_index(drop=True)
    return df_all


# ----------------- 原文读取与 speaker 解析 -----------------
def guess_raw_txt_path(doc_id: str) -> Optional[str]:
    for tdir in list_type_dirs(RAW_TXT_ROOT):
        cand = glob.glob(os.path.join(tdir, f"{doc_id}.txt"))
        if cand: return cand[0]
        prefix = doc_id.split("_")[0]
        cand = glob.glob(os.path.join(tdir, f"{prefix}*.txt"))
        if cand: return cand[0]
    return None

def normalize_role(tok: str) -> str:
    tl = (tok or "").lower().strip()
    alias = {
        "cfo":"CFO","chief financial officer":"CFO",
        "ceo":"CEO","chief executive officer":"CEO",
        "coo":"COO","cro":"CRO","cto":"CTO","cio":"CIO","cmo":"CMO",
        "svp":"SVP","senior vice president":"SVP",
        "evp":"EVP","executive vice president":"EVP",
        "vp":"VP","vice president":"VP",
        "ir":"IR","investor relations":"IR",
        "operator":"Operator","moderator":"Operator","host":"Operator",
        "analyst":"Analyst","chairman":"Chair","chair":"Chair","president":"President",
        "treasurer":"Treasurer","controller":"Controller","director":"Director",
        "general counsel":"General Counsel","managing director":"Managing Director","head":"Head",
    }
    if tl in alias: return alias[tl]
    if "chief" in tl and "financial" in tl: return "CFO"
    if "chief" in tl and "executive" in tl: return "CEO"
    if "operating officer" in tl: return "COO"
    if "information officer" in tl: return "CIO"
    if "technology officer" in tl: return "CTO"
    if "marketing officer" in tl: return "CMO"
    if "senior vice president" in tl: return "SVP"
    if "executive vice president" in tl: return "EVP"
    if "vice president" in tl: return "VP"
    if "investor relations" in tl: return "IR"
    if "analyst" in tl: return "Analyst"
    if tl in ["q&a","qa","question","questions"]: return "Analyst"
    return " ".join(w.capitalize() for w in re.split(r"\s+", (tok or "").strip()) if w)

def parse_participants_header(lines: List[str]) -> Dict[str, str]:
    name2role: Dict[str,str] = {}
    in_part = False
    for raw in lines:
        line = (raw or "").strip()
        if not line: continue
        if re.match(r"(?i)^participants\s*$", line):
            in_part = True; continue
        if in_part and re.match(r"(?i)^(presentation|prepared remarks|operator|conference call|q&a|question)", line):
            break
        if in_part:
            s = re.sub(r"\s*\[\d+\]\s*$", "", line)
            if " - " in s: left, right = s.split(" - ",1)
            elif " — " in s: left, right = s.split(" — ",1)
            elif ", " in s: left, right = s.split(", ",1)
            else: continue
            name = re.sub(r"\s+"," ", left).strip()
            role = right
            if re.search(r"(?i)\banalyst\b", role): role = "Analyst"
            role = normalize_role(role)
            if name: name2role[name.lower()] = role
    return name2role

def extract_switch_role(line: str) -> Tuple[Optional[str], Optional[str]]:
    s = re.sub(r"\s*\[\d+\]\s*$","", (line or "").strip())
    if not s: return None, None
    if re.fullmatch(r"(?i)operator|moderator|host", s): return None, "Operator"
    if re.match(r"(?i)^(q:|question:|questions?:)$", s): return None, "Analyst"
    if re.match(r"^[A-Z][A-Za-z\.\-'\s]+:\s*$", s): return s[:-1].strip(), None
    for sep in [" - ", " — ", ", "]:
        if sep in s:
            parts = [p.strip() for p in s.split(sep)]
            if len(parts)>=2:
                name = parts[0]
                tail = parts[-1]
                role = "Analyst" if re.search(r"(?i)\banalyst\b", tail) else tail
                return name, normalize_role(role)
    if len(s.split())<=6:
        if re.search(r"(?i)(chief|officer|cfo|ceo|coo|ir|analyst|vice|president|svp|evp|director|treasurer|controller)", s):
            return None, normalize_role(s)
        return s, None
    return None, None

def naive_sentence_split(text: str) -> List[str]:
    placeholders = {"e.g.":"eg_ph","i.e.":"ie_ph","U.S.":"US_ph","U.K.":"UK_ph","Mr.":"Mr_ph","Ms.":"Ms_ph"}
    tmp = text
    for k,v in placeholders.items(): tmp = tmp.replace(k,v)
    parts = re.split(r"(?<=[。！？!?\.])\s+", tmp)
    out = []
    for p in parts:
        if not p: continue
        for k,v in placeholders.items(): p = p.replace(v,k)
        p = p.strip()
        if len(p) < 2: continue
        out.append(p)
    return out

def sentence_roles_from_raw(doc_path: str) -> List[str]:
    with open(doc_path,"r",encoding="utf-8",errors="ignore") as f:
        lines = f.read().splitlines()
    name2role = parse_participants_header(lines)
    known = list(name2role.keys())

    def map_name_to_role(name_raw: str) -> Optional[str]:
        if not name_raw: return None
        key = name_raw.lower().strip()
        if key in name2role: return name2role[key]
        if known:
            mch = difflib.get_close_matches(key, known, n=1, cutoff=0.75)
            if mch: return name2role[mch[0]]
        return None

    current_role = "Unknown"
    buf, para_role_pairs = [], []

    def flush():
        nonlocal buf, current_role, para_role_pairs
        if not buf: return
        text = " ".join(buf).strip()
        if text: para_role_pairs.append((text,current_role))
        buf = []

    sep_pat = re.compile(r"^\s*[-=]{3,}\s*$")
    junk = [re.compile(p, re.I) for p in [
        r"^\s*questions? and answers?\s*$",
        r"^\s*\(operator instructions\)\s*$",
        r"^\s*forward[- ]looking statements.*$",
        r"^\s*copyright.*$",
        r"^\s*participants\s*$",
        r"^\s*presentation\s*$",
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
                r = map_name_to_role(name)
                if r: current_role = r
                elif role: current_role = role
                else: current_role = "Unknown"
            else:
                current_role = role or "Unknown"
            continue

        buf.append(line)
        if re.search(r"[。\.!?]$", line) and sum(len(x) for x in buf) > 200:
            flush()
    flush()

    roles = []
    for para, role in para_role_pairs:
        sents = naive_sentence_split(para)
        if not sents: continue
        role_here = role
        for s in sents:
            if re.match(r"(?i)^(q:|question:)\s*", s.strip()):
                role_here = "Analyst"
                continue
            roles.append(role_here)
    return roles

def attach_speaker_roles(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for doc_id, g in df.groupby("doc_id"):
        rawp = guess_raw_txt_path(doc_id)
        roles = []
        if rawp and os.path.isfile(rawp):
            try:
                roles = sentence_roles_from_raw(rawp)
            except Exception:
                roles = []
        n = len(g)
        if not roles: roles = ["Unknown"] * n
        if len(roles) < n: roles = roles + ["Unknown"]*(n-len(roles))
        elif len(roles) > n: roles = roles[:n]

        # 前向填充
        fixed, last = [], "Unknown"
        for r in roles:
            if r!="Unknown": last = r
            fixed.append(last)

        gg = g.sort_values(["sent_idx"]).copy()
        gg["speaker_role"] = fixed[:len(gg)]
        out.append(gg)

    return pd.concat(out, ignore_index=True)


# ----------------- 上下文输入 -----------------
def build_context_inputs(df: pd.DataFrame, k: int = 2) -> pd.DataFrame:
    df = df.sort_values(["doc_id","sent_idx"]).reset_index(drop=True)
    rows = []
    for doc_id, g in df.groupby("doc_id"):
        g = g.reset_index(drop=True)
        texts = g["text"].astype(str).tolist()
        roles  = g["speaker_role"].fillna("Unknown").tolist()
        for i in range(len(g)):
            left  = texts[max(0, i-k): i]
            right = texts[i+1: i+1+k]
            cur   = texts[i]
            role  = roles[i]
            inp = f"[SPEAKER={role}] "
            if left:  inp += " ".join(left) + " [CTX_BEFORE] "
            inp += cur
            if right: inp += " [CTX_AFTER] " + " ".join(right)
            row = g.loc[i].to_dict()
            row["input_with_ctx"] = inp
            rows.append(row)
    return pd.DataFrame(rows)


# ----------------- MLM 继续预训练 -----------------
class MLMDataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 320):
        self.texts = texts
        self.tok = tokenizer
        self.max_length = max_length
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

    return {
        "input_ids": ids_masked.to(device),
        "attention_mask": attn_masks.to(device),
        "labels": labels.to(device),
    }

def run_mlm_pretrain(base_model: str, texts: List[str], save_dir: str,
                     epochs=1, bs=16, lr=5e-5, mask_prob=0.15, device=None):
    """
    使用 build_context_inputs 生成的 input_with_ctx 进行 MLM 继续预训练。
    """
    base = resolve_model_path(base_model, prefer_mlm=False)
    tokenizer = AutoTokenizer.from_pretrained(base, use_fast=True, local_files_only=OFFLINE)

    special_tokens = ["[CTX_BEFORE]", "[CTX_AFTER]"] + \
                     [f"[SPEAKER={r}]" for r in [
                         "CEO","CFO","IR","Analyst","Operator","Other","Unknown",
                         "President","Chair","Director","SVP","EVP","VP"
                     ]]
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
    scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    model.train()
    for ep in range(1, epochs+1):
        total_loss = 0.0
        for step, batch in enumerate(dl, start=1):
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                out = model(**batch)
                loss = out.loss

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            scheduler.step()

            total_loss += float(loss.item())
            if step % 200 == 0:
                print(f"[MLM ep {ep} step {step}] loss={total_loss/step:.4f}")

        print(f"[MLM] epoch {ep} avg_loss={total_loss/max(1,len(dl)):.4f}")

    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"[MLM] saved checkpoint -> {save_dir}")
    return save_dir


# ----------------- 主流程（只保留 1/2/3） -----------------
def main():
    print("[Paths]")
    print("  PROJ_ROOT        :", PROJ_ROOT)
    print("  WORDBAG_OUT_ROOT :", WORDBAG_OUT_ROOT)
    print("  RAW_TXT_ROOT     :", RAW_TXT_ROOT)
    print("  CKPT_DIR         :", CKPT_DIR)
    print("  OUT_DIR          :", OUT_DIR)
    print("  MODEL_NAME       :", MODEL_NAME)

    set_offline_env()
    set_seed(SEED)
    device = get_device()
    print("  DEVICE           :", device)

    # 1) 读取 & 强化 speaker 标注
    df_raw = load_all_sentence_csvs(WORDBAG_OUT_ROOT).sort_values(["doc_id","sent_idx"]).reset_index(drop=True)
    df_raw = attach_speaker_roles(df_raw)

    # 2) 上下文增强
    df_ctx = build_context_inputs(df_raw, k=CONTEXT_K)

    # 3) 可选：先“统一学习”（内存安全，离线优先本地）
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
                base_model=MODEL_NAME,
                texts=mlm_texts,
                save_dir=mlm_ckpt_dir,
                epochs=MLM_EPOCHS,
                bs=MLM_BATCH,
                lr=MLM_LR,
                mask_prob=MLM_MASK_PROB,
                device=device
            )
    else:
        init_model_path = resolve_model_path(MODEL_NAME, prefer_mlm=True)

    print(f"[DONE] ✓ MLM-only pipeline finished. ckpt -> {init_model_path}")
    print("[STOP] Only steps 1-3 are kept in this script.")


if __name__ == "__main__":
    main()
