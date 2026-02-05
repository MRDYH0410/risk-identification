# -*- coding: utf-8 -*-  # 指定文件编码为 UTF-8，避免中文注释报错
"""
lexicon_standardizer.py  # 脚本名称：词典标准化器
--------------------------------------------
作用：把不同来源的“金融风险词典/词袋”转成统一的标准格式，便于后续合并。  # 脚本用途说明
输出：标准化 CSV（列固定为 bucket, term, source, description）。  # 输出格式描述

用法示例（在终端或 PyCharm 运行）：
    python lexicon_standardizer.py \  # 调用脚本
        --lm /path/to/Loughran-McDonald_MasterDictionary_1993-2024.csv \  # LM 原始词典路径
        --flr /path/to/combined_politicaltbb_npb_finalbigrams.csv \      # FLR 主题 bigrams 路径
        --out_dir ./standardized_lexicons                                  # 标准化输出目录
"""
import os  # 操作系统路径与目录
import re  # 正则表达式，用于文本清洗
import argparse  # 命令行参数解析
import pandas as pd  # 读写 CSV、数据处理
from typing import Dict, List, Optional  # 类型注解，增强可读性与 IDE 提示


# ----------------------------
# 通用：token 归一化
# ----------------------------
def norm_token(t: str) -> str:  # 定义统一的词条清洗函数
    """
    统一的词条清洗规则：  # 文档注释解释此函数做什么
    - 小写
    - 仅保留字母/数字/连字符/空格（允许 n-gram）
    - 合并多空格
    """
    if not isinstance(t, str):  # 如果不是字符串
        return ""  # 返回空串
    t = t.lower()  # 转小写
    t = re.sub(r"[^a-z0-9\- ]+", " ", t)  # 非字母数字/连字符/空格的字符替换为空格
    t = re.sub(r"\s+", " ", t).strip()  # 多个空格合并为一个，并去两端空格
    return t  # 返回清洗后的 token


# ----------------------------
# 适配器 1：LM Master Dictionary -> 标准化
# ----------------------------
def standardize_lm(
    lm_path: str,
    out_csv_path: str,
    lm_bucket_cols: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    读取 Loughran–McDonald Master Dictionary，并将不确定性/诉讼/约束/强弱情态/负向 等词条
    标准化为 (bucket, term, source, description) 形式。对列名和取值做鲁棒处理。
    """
    # 1) 读入（容错编码）
    enc_used = None
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            lm = pd.read_csv(lm_path, encoding=enc)
            enc_used = enc
            break
        except Exception:
            pass
    if enc_used is None:
        lm = pd.read_csv(lm_path, engine="python")
        enc_used = "python-default"

    # 2) 统一列名（小写 + 去空格 + 去BOM）
    def _norm_col(c: str) -> str:
        c = (c or "").replace("\ufeff", "").strip().lower()
        c = re.sub(r"\s+", "_", c)  # 空格换下划线
        return c
    lm.columns = [_norm_col(c) for c in lm.columns]

    # 3) 找到“单词列”（优先叫 word，其次第一列）
    word_col = "word" if "word" in lm.columns else lm.columns[0]
    lm["term"] = lm[word_col].astype(str).map(norm_token)
    lm = lm[lm["term"].str.len() > 0].copy()

    # 4) 针对不同版本的列名做“模糊匹配映射”
    #    你看到的版本里 "Positive" 写成了 "Postive"，这里统一处理。
    #    字典 key = 我们期望的逻辑名，val = 可能的原始列名关键词（小写）
    col_candidates = {
        "uncertainty":   ["uncertainty"],
        "litigious":     ["litigious", "legal"],
        "constraining":  ["constraining", "constraint", "compliance"],
        "strong_modal":  ["strong_modal", "strongmodal", "modal_strong"],
        "weak_modal":    ["weak_modal", "weakmodal", "modal_weak"],
        "negative":      ["negative"],
        # 如需 Positive，可加： "positive": ["positive", "postive"]
    }

    # 根据 candidates 在 lm.columns 中“就近匹配”真实列名
    resolved_cols: Dict[str, str] = {}
    for logical_name, keys in col_candidates.items():
        hit = None
        for k in keys:
            # 完全匹配优先
            if k in lm.columns:
                hit = k
                break
        if hit is None:
            # 退一步：包含匹配（如 'postive' vs 'positive'）
            for c in lm.columns:
                if any(k in c for k in keys):
                    hit = c
                    break
        if hit is not None:
            resolved_cols[logical_name] = hit

    # 5) 若未显式传入 bucket 映射，则用我们解析出的列
    if lm_bucket_cols is None:
        # bucket 名你可以按论文需要定制；右侧是真实列名（已统一为小写+下划线）
        lm_bucket_cols = {
            "lm_uncertainty":   resolved_cols.get("uncertainty",   None),
            "lm_litigious":     resolved_cols.get("litigious",     None),
            "lm_constraining":  resolved_cols.get("constraining",  None),
            "lm_modal_strong":  resolved_cols.get("strong_modal",  None),
            "lm_modal_weak":    resolved_cols.get("weak_modal",    None),
            "lm_negative":      resolved_cols.get("negative",      None),
        }
        # 去掉未命中的 bucket
        lm_bucket_cols = {k: v for k, v in lm_bucket_cols.items() if v is not None}

    # 6) 把这些列都转为数值（有的版本可能是字符串/float）
    for col in lm_bucket_cols.values():
        lm[col] = pd.to_numeric(lm[col], errors="coerce").fillna(0)

    # 7) 统计每个 bucket 命中词量，并输出提示
    print("[LM] column mapping:", lm_bucket_cols)
    for bkt, col in lm_bucket_cols.items():
        cnt = int((lm[col] > 0).sum())
        print(f"    {bkt:18s} <- {col:20s}  hits: {cnt}")

    # 8) 生成标准化行
    desc_map = {
        "lm_uncertainty":   "LM uncertainty words（不确定性）",
        "lm_litigious":     "LM litigious/legal words（诉讼/法律风险）",
        "lm_constraining":  "LM constraining/compliance words（约束/合规）",
        "lm_modal_strong":  "LM strong modal words（强情态/承诺强）",
        "lm_modal_weak":    "LM weak modal words（弱情态/模糊/对冲）",
        "lm_negative":      "LM negative polarity words（负向情感）",
    }

    rows = []
    for bucket, col in lm_bucket_cols.items():
        # 用 >0 作为“在该类别内”的判定（比 ==1 更稳健）
        terms = lm.loc[lm[col] > 0, "term"].dropna().astype(str).unique().tolist()
        for t in terms:
            rows.append({
                "bucket": bucket,
                "term": t,
                "source": "LM",
                "description": desc_map.get(bucket, bucket),
            })

    df_std = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df_std.to_csv(out_csv_path, index=False)
    print(f"[LM] standardized -> {out_csv_path} (rows={len(df_std)})")
    return df_std



# ----------------------------
# 适配器 2：Firm-Level Risk bigrams -> 标准化
# ----------------------------
def _map_flr_theme(raw: str) -> str:  # 将 FLR 原始标签映射到标准主题 bucket
    s = (raw or "").lower()  # 安全转换小写字符串
    if any(k in s for k in ["covid", "pandemic", "sars", "ebola", "zika", "health", "h1n1"]):
        return "flr_pandemic_health"  # 疫情/健康
    if any(k in s for k in ["policy", "polit", "govern", "regul", "tariff", "sanction", "brexit", "election"]):
        return "flr_policy_political"  # 政策/政治/制裁/关税/选举
    if any(k in s for k in ["trade", "supply", "logistic", "port", "shipment", "export", "import"]):
        return "flr_trade_supplychain"  # 贸易/供应链/物流
    if "tax" in s:
        return "flr_taxation"  # 税收
    if any(k in s for k in ["security", "war", "conflict", "geopolit"]):
        return "flr_security_geopolitical"  # 地缘/冲突/安全
    if any(k in s for k in ["climate", "carbon", "emission", "environment", "esg", "sustainab"]):
        return "flr_environment_climate"  # 气候/环境/ESG
    if any(k in s for k in ["technology", "tech", "cyber", "privacy", "finance_standard_wordbag", "ip", "ai"]):
        return "flr_technology_cyber"  # 科技/网络/隐私/IP
    if any(k in s for k in ["institution", "central bank", "fed", "ecb", "liquidity", "banking", "credit"]):
        return "flr_institutions_finance"  # 机构/央行/流动性/信用
    return "flr_external_other"  # 其他外部主题


def standardize_flr(
    flr_path: str,  # FLR 原始 CSV 路径
    out_csv_path: str,  # 标准化输出 CSV 路径
    term_col_guess: Optional[List[str]] = None,  # 可能的术语列名候选
    label_col_guess: Optional[List[str]] = None,  # 可能的标签列名候选
) -> pd.DataFrame:  # 返回标准化后的 DataFrame
    """
    读取 Firm-Level Risk 的主题 bigrams（或类似文件），  # 功能说明
    自动找出 bigram 列与标签列，映射到标准 bucket，并输出标准化 CSV。  # 自动列识别与映射
    """
    enc_used = None  # 初始化编码记录
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:  # 尝试常见编码
        try:
            flr = pd.read_csv(flr_path, encoding=enc)  # 读 CSV
            enc_used = enc  # 记录编码
            break  # 成功则退出
        except Exception:
            pass  # 失败则继续
    if enc_used is None:  # 如果都失败
        flr = pd.read_csv(flr_path, engine="python")  # 使用 python 引擎兜底
        enc_used = "python-default"  # 记录兜底编码

    cols_low = [c.lower() for c in flr.columns]  # 全部列名小写列表

    term_col = None  # 初始化术语列名
    for cand in (term_col_guess or ["bigram", "term", "token", "phrase", "ngram", "ngram_bigram", "bi_gram", "text"]):
        if cand in cols_low:  # 如果候选名在列中
            term_col = flr.columns[cols_low.index(cand)]  # 获取原始列名
            break  # 找到即停止
    if term_col is None:  # 如果没有找到术语列
        obj_cols = [c for c in flr.columns if flr[c].dtype == "object"]  # 所有对象类型列
        term_col = obj_cols[0] if obj_cols else flr.columns[0]  # 退化选择第一列

    label_col = None  # 初始化标签列名
    for cand in (label_col_guess or ["category", "label", "theme", "topic", "risk_type", "class", "group"]):
        if cand in cols_low:  # 如果候选标签名在列中
            label_col = flr.columns[cols_low.index(cand)]  # 获取原始列名
            break  # 找到即停止

    flr["term"] = flr[term_col].astype(str).map(norm_token)  # 清洗 bigram/术语
    flr = flr[flr["term"].str.len() > 0].copy()  # 仅保留非空术语

    flr["label_raw"] = flr[label_col].astype(str) if label_col else "external_risk"  # 若无标签列则给默认
    flr["bucket"] = flr["label_raw"].map(_map_flr_theme)  # 映射到标准主题 bucket

    desc_map = {  # 每个 FLR bucket 的说明
        "flr_pandemic_health":      "FLR: pandemic/health themes（疫情/健康）",
        "flr_policy_political":     "FLR: policy/political themes（政策/政治/制裁/关税/选举）",
        "flr_trade_supplychain":    "FLR: trade/supply-chain/logistics themes（贸易/供应链/物流）",
        "flr_taxation":             "FLR: taxation themes（税务）",
        "flr_security_geopolitical":"FLR: geopolitical/security themes（地缘/安全/冲突）",
        "flr_environment_climate":  "FLR: environment/climate/ESG themes（环境/气候/ESG）",
        "flr_technology_cyber":     "FLR: technology/cyber/privacy/IP themes（科技/网络/隐私/IP）",
        "flr_institutions_finance": "FLR: institutions/central-bank/liquidity/credit（机构/央行/流动性/信用）",
        "flr_external_other":       "FLR: other external themes（其他外部主题）",
    }

    rows = []  # 初始化输出行
    for bkt, grp in flr.groupby("bucket"):  # 按 bucket 分组
        terms = sorted(set(grp["term"].dropna().tolist()))  # 获取唯一术语集合
        for t in terms:  # 遍历术语
            rows.append({  # 追加标准化记录
                "bucket": bkt,  # 风险类别
                "term": t,  # 术语
                "source": "FLR",  # 来源 FLR
                "description": desc_map.get(bkt, bkt)  # 描述
            })

    df_std = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)  # 构造 DataFrame 并去重
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)  # 创建输出目录
    df_std.to_csv(out_csv_path, index=False)  # 写出 CSV
    print(f"[FLR] standardized -> {out_csv_path} (rows={len(df_std)})")  # 控制台提示
    return df_std  # 返回标准化结果


# ----------------------------
# CLI 主入口
# ----------------------------
def main():  # 定义脚本主函数
    out_dir = "./standardized_lexicons"                 # 输出目录
    os.makedirs(out_dir, exist_ok=True)                 # 确保目录存在

    lm_path  = "finance_standard_wordbag/Loughran-McDonald_MasterDictionary_1993-2024.csv"  # LM 输入
    flr_path = "finance_standard_wordbag/combined_politicaltbb_npb_finalbigrams.csv"  # FLR 输入

    # 路径建议使用正斜杠或原始字符串 r"..."
    standardize_lm(lm_path,  os.path.join(out_dir, "lm_standardized.csv"))
    standardize_flr(flr_path, os.path.join(out_dir, "flr_standardized.csv"))


if __name__ == "__main__":  # 如果作为脚本运行
    main()  # 调用主函数
