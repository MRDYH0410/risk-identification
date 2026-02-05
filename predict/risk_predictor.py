# -*- coding: utf-8 -*-
"""
risk_predictor.py —— LGBM / XGB / Ensemble（按类偏好权重搜索 + 阈值优化）
+ NEW: 在 Ensemble 上训练“解释层”多项逻辑回归（surrogate），并输出：
    - output/coef_multinomial_long.csv
    - output/preds_multinomial.csv
用于 attribution.py 的 a_mij -> y_mz 归因（基于 Ensemble 决策）。

本版本改动（按你的要求）：
- 所有绘图相关代码移除（不再生成 png）
- 仅导出用于 predictor_display.py 出图的必要文件：
  output/verification/preds_val_lgbm.csv
  output/verification/preds_val_xgb.csv
  output/verification/preds_val_ens.csv
  以及混淆矩阵 CSV、classification_report TXT、metrics JSON、ensemble_tuning_trend.csv、对比指标 CSV
"""

import os
import re
import json
import shutil
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report,
    roc_curve, auc, average_precision_score, precision_recall_curve,
    brier_score_loss
)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# ============ 可选：LightGBM ============
try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False

# ============ 可选：XGBoost ============
HAS_XGB = False
XGB_HAS_EARLYSTOP_CB = False
try:
    import xgboost as xgb
    from xgboost import XGBClassifier
    HAS_XGB = True
    try:
        _ = xgb.callback.EarlyStopping
        XGB_HAS_EARLYSTOP_CB = True
    except Exception:
        XGB_HAS_EARLYSTOP_CB = False
except Exception:
    HAS_XGB = False


# -------------------- 路径 --------------------
BASE_DIR  = os.path.dirname(__file__)                    # .../predict
PROJ_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

IN_DIR     = os.path.join(PROJ_ROOT, "identification", "output")
OUT_DIR    = os.path.join(BASE_DIR, "output")
LOG_DIR    = os.path.join(OUT_DIR, "logs")
VERIF_DIR  = os.path.join(OUT_DIR, "verification")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(VERIF_DIR, exist_ok=True)

A_MIJ_CSV = os.path.join(IN_DIR, "a_mij__sentence_ctx.csv")
SENT_CSV  = os.path.join(IN_DIR, "sentences_scored_llm_ctx.csv")


# -------------------- 常量/超参 --------------------
ALPHA_SMOOTH   = 1.0
MIN_SENT_FOR_F = 5
TAU            = 1.5
RANDOM_SEED    = 42

THRESH_GRID = np.linspace(0.2, 0.8, 13)

# —— 融合权重网格（按类偏好）——
# 记：P_ens[:,z] = w_z * P_lgb[:,z] + (1 - w_z) * P_xgb[:,z]
BLEND_WEIGHT_GRID_BY_CLASS = {
    1: np.array([0.00, 0.10, 0.25], dtype=float),  # 更靠 XGB
    2: np.array([0.75, 0.90, 1.00], dtype=float),  # 更靠 LGB
    3: np.array([0.25, 0.50, 0.75], dtype=float)   # 中性
}
BLEND_SELECT_STRATEGY = "ap"

CANON_KEYS = ["risk_uncertainty", "risk_legal", "risk_constraint", "risk_external"]
PROB_COLS  = [
    "prob_risk_uncertainty", "prob_risk_legal",
    "prob_risk_constraint",  "prob_risk_external"
]


# -------------------- 实用函数 --------------------
def _canon_risk_bucket(x: str) -> str:
    if not isinstance(x, str):
        return ""
    t = re.sub(r"[^a-z0-9]+", "", x.lower())
    if "uncertain" in t or "q1" in t:
        return "risk_uncertainty"
    if "legal" in t or "q2" in t:
        return "risk_legal"
    if "constraint" in t or "operational" in t or "q3" in t:
        return "risk_constraint"
    if "external" in t or "macro" in t or "q4" in t:
        return "risk_external"
    return ""

def _canon_role(x) -> str:
    return (str(x) if pd.notna(x) else "").strip()

def _canon_company(x) -> str:
    s = str(x) if pd.notna(x) else ""
    return re.sub(r"\s+", " ", s).strip()

def _extract_z_from_source_type(v: str):
    if not isinstance(v, str):
        return np.nan
    m = re.search(r"Type_([123])", v, flags=re.IGNORECASE)
    return int(m.group(1)) if m else np.nan

def _np_to_py(o):
    import numpy as _np
    if isinstance(o, (_np.integer, _np.floating)):
        return o.item()
    if isinstance(o, _np.ndarray):
        return o.tolist()
    if isinstance(o, dict):
        return {k: _np_to_py(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_np_to_py(v) for v in o]
    return o

def save_json(obj, name: str, folder: str):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_np_to_py(obj), f, ensure_ascii=False, indent=2)
    print(f"[JSON]  {path}")
    return path

def write_df(df: pd.DataFrame, name: str, folder: str):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, name)
    df.to_csv(path, index=False)
    print(f"[WRITE] {path} rows={len(df)}")
    return path

def DataFrameSafe(values, index, columns):
    try:
        return pd.DataFrame(values, index=index, columns=columns)
    except Exception:
        return pd.DataFrame(values.toarray(), index=index, columns=columns)


# -------------------- 数据读取/先验/特征 --------------------
def self_check_inputs(a_path: str, s_path: str):
    report = {}
    a = pd.read_csv(a_path, usecols=None)
    a_companies = set(_canon_company(x) for x in (a["company_id"] if "company_id" in a.columns else a[a.columns[0]]))
    report["a_mij_company_count"] = len(a_companies)

    usecols = ["company_id", "speaker_role", "source_type"]
    head = pd.read_csv(s_path, nrows=0).columns.tolist()
    for c in PROB_COLS:
        if c in head and c not in usecols:
            usecols.append(c)

    s = pd.read_csv(s_path, usecols=[c for c in usecols if c in head])
    s["company_id"] = s["company_id"].map(_canon_company)
    s["z_label"] = s["source_type"].map(_extract_z_from_source_type)
    s = s[s["z_label"].notna()].copy()
    s["z_label"] = s["z_label"].astype(int)

    s_companies = set(s["company_id"].unique())
    report["sentences_company_count_total"] = len(s_companies)
    report["sentences_company_count_by_z"] = s.groupby("z_label")["company_id"].nunique().to_dict()
    report["companies_intersection"] = len(a_companies & s_companies)
    report["companies_union"] = len(a_companies | s_companies)

    save_json(report, "selfcheck_inputs.json", LOG_DIR)
    with open(os.path.join(LOG_DIR, "selfcheck_inputs.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(report, ensure_ascii=False, indent=2))
    print("[SELF-CHECK][inputs]", report)
    return report

def load_a_mij(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    c_company = cols.get("company_id") or cols.get("doc_id") or cols.get("firm_id") or cols.get("ticker")
    c_role    = cols.get("role") or cols.get("speaker_role") or cols.get("speaker")
    c_risk    = cols.get("risk_bucket") or cols.get("risk") or cols.get("risk_class")
    c_amij    = cols.get("a_mij") or cols.get("prop") or cols.get("ratio") or cols.get("share") or cols.get("pct")
    if not (c_company and c_role and c_risk and c_amij):
        raise ValueError(f"[a_mij] 缺少必要列，看到的列：{list(df.columns)}")

    out = (df[[c_company, c_role, c_risk, c_amij]].copy()
           .rename(columns={c_company: "company_id", c_role: "role",
                            c_risk: "risk_bucket", c_amij: "a_mij"}))
    out["company_id"] = out["company_id"].map(_canon_company)
    out["role"]       = out["role"].map(_canon_role)
    out["risk_bucket"] = out["risk_bucket"].map(_canon_risk_bucket)
    out["a_mij"]      = pd.to_numeric(out["a_mij"], errors="coerce").fillna(0.0)
    out = out[out["risk_bucket"].isin(CANON_KEYS)].copy()
    out.reset_index(drop=True, inplace=True)
    return out

def load_sentences(path: str):
    df_head = pd.read_csv(path, nrows=0)
    base_cols = ["company_id", "speaker_role", "source_type"]
    if "doc_id" in df_head.columns:
        base_cols.append("doc_id")
    need = base_cols + [c for c in PROB_COLS if c in df_head.columns]
    df = pd.read_csv(path, usecols=need)

    df["company_id"] = df["company_id"].map(_canon_company)
    if "doc_id" not in df.columns:
        df["doc_id"] = df["company_id"]
    df["role"]    = df["speaker_role"].map(_canon_role)
    df["z_label"] = df["source_type"].map(_extract_z_from_source_type)
    df = df[df["z_label"].notna()].copy()
    df["z_label"] = df["z_label"].astype(int)

    long = df[["company_id", "doc_id", "role", "z_label"] + [c for c in PROB_COLS if c in df.columns]].melt(
        id_vars=["company_id", "doc_id", "role", "z_label"],
        value_vars=[c for c in PROB_COLS if c in df.columns],
        var_name="prob_key", value_name="s_val"
    )
    key2bucket = {}
    for k in PROB_COLS:
        for std in CANON_KEYS:
            if std in k:
                key2bucket[k] = std
    long["risk_bucket"] = long["prob_key"].map(key2bucket).fillna("")
    long = long[long["risk_bucket"].isin(CANON_KEYS)].copy()
    long["s_val"] = pd.to_numeric(long["s_val"], errors="coerce").fillna(0.0)

    pool = (long.groupby(["role", "risk_bucket", "z_label"], as_index=False)
            .agg(sum_s=("s_val", "sum"),
                 mean_s=("s_val", "mean"),
                 N_sent=("s_val", "size")))
    per_m = (long.groupby(["company_id", "doc_id", "role", "risk_bucket", "z_label"], as_index=False)
             .agg(sum_s=("s_val", "sum"),
                  mean_s=("s_val", "mean"),
                  N_sent=("s_val", "size")))
    y_by_m = (df.groupby("company_id")["z_label"]
              .agg(lambda s: s.value_counts().idxmax())
              .rename("y_true").reset_index())
    return pool, per_m, y_by_m

def estimate_fijz(pool_df: pd.DataFrame, alpha: float = 1.0, min_sent: int = 5) -> pd.DataFrame:
    totals = (pool_df.groupby(["role", "risk_bucket"], as_index=False)
              .agg(total_s=("sum_s", "sum"),
                   total_n=("N_sent", "sum")))
    m = pd.merge(pool_df, totals, on=["role", "risk_bucket"], how="left")
    m["other_s"] = (m["total_s"] - m["sum_s"]).clip(lower=0.0)
    m["other_n"] = (m["total_n"] - m["N_sent"]).clip(lower=0)
    adj_alpha = alpha * (1.0 + (min_sent / (m["N_sent"].replace(0, 1))).clip(upper=10.0))
    num = m["sum_s"] + adj_alpha
    den = m["other_s"] + adj_alpha
    m["f_ijz"] = np.log(num / den)
    out = m[["role", "risk_bucket", "z_label", "f_ijz", "sum_s", "N_sent"]].copy()
    out = out.sort_values(["role", "risk_bucket", "z_label"]).reset_index(drop=True)
    return out

def make_company_features(a_mij: pd.DataFrame, fijz: pd.DataFrame):
    a_mij = a_mij.copy()
    a_mij["role_risk"] = a_mij["role"].astype(str) + "|" + a_mij["risk_bucket"].astype(str)
    X_base = a_mij.pivot_table(
        index="company_id",
        columns="role_risk",
        values="a_mij",
        aggfunc="sum"
    ).fillna(0.0)

    priors = {}
    for z in [1, 2, 3]:
        fz = fijz[fijz["z_label"] == z][["role", "risk_bucket", "f_ijz", "N_sent"]].copy()
        if fz.empty:
            priors[f"prior_z{z}"] = pd.Series(0.0, index=X_base.index)
            continue
        fz["role_risk"] = fz["role"].astype(str) + "|" + fz["risk_bucket"].astype(str)
        w = np.log1p(fz["N_sent"].clip(lower=1))
        f_vec = (fz["f_ijz"] / TAU) * (w / (w.max() + 1e-6))
        f_vec = f_vec.groupby(fz["role_risk"]).sum()
        f_aligned = f_vec.reindex(X_base.columns).fillna(0.0).values
        priors[f"prior_z{z}"] = pd.Series((X_base.values @ f_aligned), index=X_base.index)

    prior_df = pd.concat(priors, axis=1)
    features = pd.concat([X_base, prior_df], axis=1).fillna(0.0)

    scaler = StandardScaler(with_mean=False)
    features_scaled = DataFrameSafe(
        scaler.fit_transform(features),
        index=features.index,
        columns=features.columns
    )
    return features_scaled, X_base, prior_df


# -------------------- LightGBM / XGBoost 训练 --------------------
def _lgb_early_stopping_cb(stopping_rounds=120):
    try:
        return lgb.early_stopping(stopping_rounds=stopping_rounds, verbose=False)
    except Exception:
        return lgb.early_stopping(stopping_rounds=stopping_rounds)

def fit_lgbm_multiclass(X, y, train_ids, val_ids):
    if not HAS_LGBM:
        raise RuntimeError("LightGBM 未安装。")

    y_map = y.set_index("company_id")["y_true"].astype(int)
    X_tr, X_va = X.loc[train_ids], X.loc[val_ids]
    y_tr, y_va = y_map.reindex(train_ids).values, y_map.reindex(val_ids).values

    class_counts = pd.Series(y_tr).value_counts().to_dict()
    total = len(y_tr)
    cls_w = {c: total / (len(class_counts) * class_counts[c]) for c in class_counts}
    w_tr = np.array([cls_w.get(int(c), 1.0) for c in y_tr])

    params = dict(
        objective="multiclass",
        num_class=3,
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=1,
        min_data_in_leaf=10,
        metric="None",
        verbose=-1,
        seed=RANDOM_SEED
    )
    dtr = lgb.Dataset(X_tr.values, label=y_tr - 1, weight=w_tr)
    dva = lgb.Dataset(X_va.values, label=y_va - 1, reference=dtr)

    def feval_macro_f1(y_pred, dset):
        yt = dset.get_label().astype(int)
        ph = y_pred.reshape(-1, 3).argmax(1)
        return "macro_f1", f1_score(yt, ph, average="macro"), True

    callbacks = [_lgb_early_stopping_cb(120)]
    try:
        model = lgb.train(
            params, dtr,
            valid_sets=[dtr, dva], valid_names=["train", "val"],
            num_boost_round=2000, feval=feval_macro_f1,
            callbacks=callbacks
        )
    except TypeError:
        model = lgb.train(
            params, dtr,
            valid_sets=[dtr, dva], valid_names=["train", "val"],
            num_boost_round=500, feval=feval_macro_f1
        )

    best_iter = getattr(model, "best_iteration", None)
    if best_iter is None or best_iter == 0:
        best_iter = getattr(model, "current_iteration", None) or 0

    prob_tr = model.predict(X_tr.values, num_iteration=best_iter if best_iter else None)
    prob_va = model.predict(X_va.values, num_iteration=best_iter if best_iter else None)
    pred_tr = prob_tr.argmax(1) + 1
    pred_va = prob_va.argmax(1) + 1

    metrics = {
        "train_acc": accuracy_score(y_tr, pred_tr),
        "train_f1":  f1_score(y_tr, pred_tr, average="macro"),
        "val_acc":   accuracy_score(y_va, pred_va),
        "val_f1":    f1_score(y_va, pred_va, average="macro"),
        "best_iter": int(best_iter or 0)
    }
    return model, (y_tr, pred_tr, prob_tr), (y_va, pred_va, prob_va), metrics

def _xgb_get_best_iter(model):
    for attr in ["best_iteration", "best_iteration_", "best_ntree_limit"]:
        if hasattr(model, attr) and getattr(model, attr):
            return int(getattr(model, attr))
    try:
        bst = model.get_booster()
        if hasattr(bst, "best_iteration") and bst.best_iteration is not None:
            return int(bst.best_iteration)
        if hasattr(bst, "best_ntree_limit") and bst.best_ntree_limit is not None:
            return int(bst.best_ntree_limit)
    except Exception:
        pass
    return 0

def fit_xgb_multiclass(X, y, train_ids, val_ids):
    if not HAS_XGB:
        raise RuntimeError("XGBoost 未安装。")

    y_map = y.set_index("company_id")["y_true"].astype(int)
    X_tr_full, X_va = X.loc[train_ids], X.loc[val_ids]
    y_tr_full, y_va = y_map.reindex(train_ids).values, y_map.reindex(val_ids).values

    X_tr, X_es, y_tr, y_es = train_test_split(
        X_tr_full.values, y_tr_full,
        test_size=0.1, random_state=RANDOM_SEED, stratify=y_tr_full
    )

    class_counts = pd.Series(y_tr).value_counts().to_dict()
    total = len(y_tr)
    cls_w = {c: total / (len(class_counts) * class_counts[c]) for c in class_counts}
    w_tr = np.array([cls_w.get(int(c), 1.0) for c in y_tr])

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.8,
        n_estimators=3000,
        eval_metric="mlogloss",
        random_state=RANDOM_SEED,
        n_jobs=-1
    )

    base_fit = dict(
        X=X_tr, y=y_tr - 1,
        sample_weight=w_tr,
        eval_set=[(X_tr, y_tr - 1), (X_es, y_es - 1)],
        verbose=False
    )

    used_callback = False
    if XGB_HAS_EARLYSTOP_CB:
        try:
            es_cb = xgb.callback.EarlyStopping(rounds=120, save_best=True, maximize=False)
            model.fit(**base_fit, callbacks=[es_cb])
            used_callback = True
        except TypeError:
            used_callback = False

    if not used_callback:
        try:
            model.fit(**base_fit, early_stopping_rounds=120)
        except TypeError:
            model.fit(**base_fit)

    prob_tr = model.predict_proba(X_tr_full.values)
    prob_va = model.predict_proba(X_va.values)
    pred_tr = prob_tr.argmax(1) + 1
    pred_va = prob_va.argmax(1) + 1

    best_iter = _xgb_get_best_iter(model)
    metrics = {
        "train_acc": accuracy_score(y_tr_full, pred_tr),
        "train_f1":  f1_score(y_tr_full, pred_tr, average="macro"),
        "val_acc":   accuracy_score(y_va, pred_va),
        "val_f1":    f1_score(y_va, pred_va, average="macro"),
        "best_iter": int(best_iter or 0)
    }
    return model, (y_tr_full, pred_tr, prob_tr), (y_va, pred_va, prob_va), metrics


# -------------------- 阈值优化（逐类） --------------------
def optimize_thresholds(y_true, proba, grid=THRESH_GRID):
    y_true = np.asarray(y_true).astype(int)
    P = np.asarray(proba, dtype=float)
    best = (-1.0, (0.5, 0.5, 0.5), None)
    cls_idx = np.array([1, 2, 3], dtype=int)

    for t1 in grid:
        for t2 in grid:
            for t3 in grid:
                th = np.array([t1, t2, t3])[None, :]
                over = (P >= th)
                cls_over = (over * cls_idx[None, :]).max(axis=1)
                argmax = P.argmax(axis=1) + 1
                has_any = over.any(axis=1)
                preds = np.where(has_any, cls_over, argmax)
                f1 = f1_score(y_true, preds, average="macro")
                if f1 > best[0]:
                    best = (f1, (float(t1), float(t2), float(t3)), preds)

    return np.array(best[1]), best[2], float(best[0])


# -------------------- 评估：更能拉开差异的指标 --------------------
def compute_auprc_auc_per_class(y_true, proba):
    y_true = np.asarray(y_true).astype(int)
    P = np.asarray(proba, dtype=float)
    ap_list, auc_list = [], []

    for z in [1, 2, 3]:
        yb = (y_true == z).astype(int)
        pz = P[:, z - 1]
        fpr, tpr, _ = roc_curve(yb, pz)
        ap_list.append(average_precision_score(yb, pz))
        auc_list.append(auc(fpr, tpr))

    return {
        "ap_z1": ap_list[0], "ap_z2": ap_list[1], "ap_z3": ap_list[2],
        "auc_z1": auc_list[0], "auc_z2": auc_list[1], "auc_z3": auc_list[2],
        "macro_auprc": float(np.mean(ap_list)),
        "macro_auc": float(np.mean(auc_list)),
        "min_auprc": float(np.min(ap_list))
    }

def compute_brier_macro(y_true, proba):
    y_true = np.asarray(y_true).astype(int)
    P = np.asarray(proba, dtype=float)
    scores = []
    for z in [1, 2, 3]:
        yb = (y_true == z).astype(int)
        pz = P[:, z - 1]
        scores.append(brier_score_loss(yb, pz))
    return float(np.mean(scores))

def recalls_per_class(y_true, preds):
    y_true = np.asarray(y_true).astype(int)
    preds = np.asarray(preds).astype(int)
    recs = []
    for z in [1, 2, 3]:
        mask = (y_true == z)
        recs.append(float((preds[mask] == z).sum() / mask.sum()) if mask.sum() else 0.0)
    return {"recall_z1": recs[0], "recall_z2": recs[1], "recall_z3": recs[2]}

def enrich_discriminative_metrics(metrics_dict, y_true, proba, preds_thresh):
    extra = compute_auprc_auc_per_class(y_true, proba)
    brier = compute_brier_macro(y_true, proba)
    recs = recalls_per_class(y_true, preds_thresh)
    metrics_dict.update(extra)
    metrics_dict["brier_macro"] = float(brier)
    metrics_dict["one_minus_brier"] = float(1.0 - brier)
    metrics_dict.update(recs)
    return metrics_dict


# -------------------- 按类加权融合（带类别偏好 + 调参轨迹） --------------------
def fast_search_blend_weights(
    y_true, proba_lgb, proba_xgb,
    weight_grid_by_class=BLEND_WEIGHT_GRID_BY_CLASS,
    thresh_grid=THRESH_GRID,
    strategy=BLEND_SELECT_STRATEGY
):
    y_true = np.asarray(y_true).astype(int)
    L = np.asarray(proba_lgb, dtype=float)
    X = np.asarray(proba_xgb, dtype=float)

    if L.shape != X.shape or L.ndim != 2 or L.shape[1] != 3:
        raise ValueError(f"[Ensemble] prob shape mismatch: L{L.shape} vs X{X.shape}")

    grids = {
        1: np.asarray(weight_grid_by_class.get(1, [0.25]), dtype=float),
        2: np.asarray(weight_grid_by_class.get(2, [0.75]), dtype=float),
        3: np.asarray(weight_grid_by_class.get(3, [0.50]), dtype=float),
    }

    best_w = np.zeros(3, dtype=float)
    for z in [1, 2, 3]:
        g = grids[z]
        best_w[z - 1] = float(g[len(g) // 2])

    history = []

    def eval_with_thresholds(w_vec):
        P = w_vec[None, :] * L + (1.0 - w_vec[None, :]) * X
        _, _, f1_macro = optimize_thresholds(y_true, P, grid=thresh_grid)
        return float(f1_macro), P

    step = 0
    for z in [1, 2, 3]:
        y_bin = (y_true == z).astype(int)
        cand_grid = grids[z]

        cur_best_local = -np.inf
        cur_best_w = best_w[z - 1]

        for w in cand_grid:
            pz = (w * L[:, z - 1]) + ((1.0 - w) * X[:, z - 1])
            if strategy == "ap":
                local_score = average_precision_score(y_bin, pz)
            else:
                fpr, tpr, _ = roc_curve(y_bin, pz)
                local_score = auc(fpr, tpr)

            tmp_w = best_w.copy()
            tmp_w[z - 1] = float(w)
            tmp_f1, _ = eval_with_thresholds(tmp_w)

            step += 1
            history.append({"step": step, "w1": tmp_w[0], "w2": tmp_w[1], "w3": tmp_w[2], "macro_f1": tmp_f1})

            if local_score > cur_best_local:
                cur_best_local = local_score
                cur_best_w = float(w)

        best_w[z - 1] = cur_best_w
        tmp_f1, _ = eval_with_thresholds(best_w)
        step += 1
        history.append({"step": step, "w1": best_w[0], "w2": best_w[1], "w3": best_w[2], "macro_f1": tmp_f1})

    P_ens = best_w[None, :] * L + (1.0 - best_w[None, :]) * X
    th, preds, f1 = optimize_thresholds(y_true, P_ens, grid=thresh_grid)
    return best_w, th, preds, float(f1), P_ens, history


# -------------------- 训练/评估主流程 --------------------
def stratified_group_split(y_df: pd.DataFrame, val_size=0.2, seed=42):
    rng = check_random_state(seed)
    y_df = y_df.copy()
    y_df["company_id"] = y_df["company_id"].astype(str)
    val_companies = []
    for z, g in y_df.groupby("y_true"):
        comps = g["company_id"].unique().tolist()
        n_val = max(1, int(round(len(comps) * val_size)))
        rng.shuffle(comps)
        val_companies += comps[:n_val]
    val_set = set(val_companies)
    train_companies = [c for c in y_df["company_id"].unique().tolist() if c not in val_set]
    return train_companies, val_companies

def self_check_preds(preds_all: pd.DataFrame, input_report: dict):
    out = {}
    out["preds_company_count"] = preds_all["company_id"].nunique()
    out["preds_by_z_star"] = preds_all["z_star"].value_counts().sort_index().to_dict()
    out["input_sentences_company_total"] = input_report.get("sentences_company_count_total", None)
    out["input_a_mij_company_total"] = input_report.get("a_mij_company_count", None)
    save_json(out, "selfcheck_inputs_vs_preds.json", LOG_DIR)
    with open(os.path.join(LOG_DIR, "selfcheck_inputs_vs_preds.txt"), "w", encoding="utf-8") as f:
        f.write(json.dumps(out, ensure_ascii=False, indent=2))
    print("[SELF-CHECK][inputs vs preds]", out)
    return out

def _export_cm_and_report(y_true, y_pred, suffix: str):
    labels = [1, 2, 3]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    write_df(pd.DataFrame(cm, index=[f"true_{z}" for z in labels],
                          columns=[f"pred_{z}" for z in labels]),
             f"confusion_matrix_val_{suffix}.csv", VERIF_DIR)
    with open(os.path.join(VERIF_DIR, f"classification_report_val_{suffix}.txt"), "w", encoding="utf-8") as f:
        f.write(classification_report(y_true, y_pred, labels=labels))

def _export_model_compare_tables(metrics_lgbm: dict, metrics_xgb: dict, metrics_ens: dict):
    if metrics_lgbm is None or metrics_xgb is None or metrics_ens is None:
        return

    rows = []
    for name, m in [("LGBM", metrics_lgbm), ("XGB", metrics_xgb), ("ENS", metrics_ens)]:
        rows.append({
            "model": name,
            "macro_auprc": m.get("macro_auprc", 0.0),
            "min_auprc": m.get("min_auprc", 0.0),
            "macro_auc": m.get("macro_auc", 0.0),
            "one_minus_brier": m.get("one_minus_brier", 0.0),
            "recall_z2": m.get("recall_z2", 0.0),
            "recall_z3": m.get("recall_z3", 0.0),
        })
    df = pd.DataFrame(rows).set_index("model")
    write_df(df.reset_index(), "model_perf_compare_enhanced_raw.csv", VERIF_DIR)

    def minmax(v):
        v = np.asarray(v, dtype=float)
        rng = float(v.max() - v.min())
        if rng <= 1e-12:
            return np.ones_like(v) * 0.5
        return (v - v.min()) / rng

    norm = pd.DataFrame({c: minmax(df[c].values) for c in df.columns}, index=df.index)
    write_df(norm.reset_index(), "model_perf_compare_enhanced.csv", VERIF_DIR)

def main():
    print("[Paths]")
    print("  IN_DIR :", IN_DIR)
    print("  OUT_DIR:", OUT_DIR)
    print("  LOG_DIR:", LOG_DIR)
    print("  VERIF  :", VERIF_DIR)
    print("  A_MIJ  :", A_MIJ_CSV)
    print("  SENT   :", SENT_CSV)

    # 0) 自检输入
    input_report = self_check_inputs(A_MIJ_CSV, SENT_CSV)

    # 1) load
    a_mij = load_a_mij(A_MIJ_CSV)
    write_df(a_mij, "a_mij_clean.csv", LOG_DIR)
    write_df(a_mij, "a_mij_clean.csv", OUT_DIR)  # 给 attribution.py

    s_pool, s_m, y_df = load_sentences(SENT_CSV)
    write_df(s_pool, "s_ijz_pool.csv", LOG_DIR)
    write_df(s_m, "s_m_ijz_agg.csv", LOG_DIR)
    write_df(y_df, "labels_y.csv", LOG_DIR)

    # 2) f_ijz
    fijz = estimate_fijz(s_pool, alpha=ALPHA_SMOOTH, min_sent=MIN_SENT_FOR_F)
    write_df(fijz, "fijz_logratio.csv", LOG_DIR)
    write_df(fijz, "fijz_logratio.csv", OUT_DIR)  # 给 attribution.py

    # 3) features
    X, X_base, pri = make_company_features(a_mij, fijz)

    write_df(X_base.reset_index().rename(columns={"index": "company_id"}), "features_X_base.csv", LOG_DIR)
    write_df(pri.reset_index().rename(columns={"index": "company_id"}), "features_prior.csv", LOG_DIR)
    write_df(X.reset_index().rename(columns={"index": "company_id"}), "features_full.csv", LOG_DIR)

    # 同步到 OUT_DIR，供 attribution.py 对齐使用
    write_df(X_base.reset_index().rename(columns={"index": "company_id"}), "features_X_base.csv", OUT_DIR)

    # 解释层用未标准化特征
    X_lin = pd.concat([X_base, pri], axis=1).fillna(0.0)
    X_lin.index = X_lin.index.astype(str)

    # 4) split
    train_ids, val_ids = stratified_group_split(y_df, val_size=0.2, seed=RANDOM_SEED)
    save_json({"train_companies": train_ids, "val_companies": val_ids}, "split_meta.json", LOG_DIR)
    print(f"[SPLIT] train={len(train_ids)} companies, val={len(val_ids)} companies")

    metrics_summary = {}
    legacy_done = False

    # ===== LightGBM =====
    df_tr_lgb, df_va_lgb = None, None
    if HAS_LGBM:
        model_lgb, (y_tr, pred_tr, prob_tr), (y_va, pred_va, prob_va), metrics_lgb = \
            fit_lgbm_multiclass(X, y_df, train_ids, val_ids)

        th_lgb, pred_va_th, f1_th_lgb = optimize_thresholds(y_va, prob_va)
        metrics_lgb["val_f1_thresh"] = float(f1_th_lgb)
        save_json({"best_thresholds": th_lgb, "val_macro_f1_thresh": f1_th_lgb}, "best_thresholds_lgbm.json", VERIF_DIR)

        metrics_lgb = enrich_discriminative_metrics(metrics_lgb, y_va, prob_va, pred_va_th)
        metrics_summary["LGBM"] = metrics_lgb
        save_json(metrics_lgb, "metrics_lgbm.json", VERIF_DIR)

        # 训练集 preds（给你留着排查）
        df_tr_lgb = pd.DataFrame({
            "company_id": train_ids,
            "y_true": y_tr,
            "y_pred_nominal": pred_tr,
            "prob_z1": prob_tr[:, 0], "prob_z2": prob_tr[:, 1], "prob_z3": prob_tr[:, 2],
        })
        write_df(df_tr_lgb, "preds_train_lgbm.csv", VERIF_DIR)

        # 验证集 preds（关键：同时存 nominal + thresh）
        df_va_lgb = pd.DataFrame({
            "company_id": val_ids,
            "y_true": y_va,
            "y_pred_nominal": pred_va,
            "y_pred_thresh": pred_va_th,
            "prob_z1": prob_va[:, 0], "prob_z2": prob_va[:, 1], "prob_z3": prob_va[:, 2],
        })
        write_df(df_va_lgb, "preds_val_lgbm.csv", VERIF_DIR)

        _export_cm_and_report(y_va, pred_va_th, "lgbm")
    else:
        print("[INFO] LightGBM 未安装，跳过 LGBM。")

    # ===== XGBoost =====
    df_tr_xgb, df_va_xgb = None, None
    if HAS_XGB:
        model_xgb, (y_tr_b, pred_tr_b, prob_tr_b), (y_va_b, pred_va_b, prob_va_b), metrics_xgb = \
            fit_xgb_multiclass(X, y_df, train_ids, val_ids)

        th_xgb, pred_va_th_b, f1_th_xgb = optimize_thresholds(y_va_b, prob_va_b)
        metrics_xgb["val_f1_thresh"] = float(f1_th_xgb)
        save_json({"best_thresholds": th_xgb, "val_macro_f1_thresh": f1_th_xgb}, "best_thresholds_xgb.json", VERIF_DIR)

        metrics_xgb = enrich_discriminative_metrics(metrics_xgb, y_va_b, prob_va_b, pred_va_th_b)
        metrics_summary["XGB"] = metrics_xgb
        save_json(metrics_xgb, "metrics_xgb.json", VERIF_DIR)

        df_tr_xgb = pd.DataFrame({
            "company_id": train_ids,
            "y_true": y_tr_b,
            "y_pred_nominal": pred_tr_b,
            "prob_z1": prob_tr_b[:, 0], "prob_z2": prob_tr_b[:, 1], "prob_z3": prob_tr_b[:, 2],
        })
        write_df(df_tr_xgb, "preds_train_xgb.csv", VERIF_DIR)

        df_va_xgb = pd.DataFrame({
            "company_id": val_ids,
            "y_true": y_va_b,
            "y_pred_nominal": pred_va_b,
            "y_pred_thresh": pred_va_th_b,
            "prob_z1": prob_va_b[:, 0], "prob_z2": prob_va_b[:, 1], "prob_z3": prob_va_b[:, 2],
        })
        write_df(df_va_xgb, "preds_val_xgb.csv", VERIF_DIR)

        _export_cm_and_report(y_va_b, pred_va_th_b, "xgb")
    else:
        print("[INFO] XGBoost 未安装，跳过 XGB。")

    # ===== Ensemble（两模型均可用） + 解释层 Surrogate LR =====
    metrics_ens = None
    preds_multinomial_written = False

    if ("LGBM" in metrics_summary) and ("XGB" in metrics_summary):
        # 从上面拿到需要的数组
        y_va = df_va_lgb["y_true"].values.astype(int)
        prob_va = df_va_lgb[["prob_z1", "prob_z2", "prob_z3"]].values.astype(float)

        y_va_b = df_va_xgb["y_true"].values.astype(int)
        prob_va_b = df_va_xgb[["prob_z1", "prob_z2", "prob_z3"]].values.astype(float)

        if not np.array_equal(y_va, y_va_b):
            print("[WARN] LGBM/XGB 的 val y_true 不一致，检查 split 或数据对齐。")

        # 1) val 上搜索按类权重 + 阈值
        w_best, th_best, pred_best, f1_best, P_ens_va, history = fast_search_blend_weights(
            y_true=y_va,
            proba_lgb=prob_va,
            proba_xgb=prob_va_b,
            weight_grid_by_class=BLEND_WEIGHT_GRID_BY_CLASS,
            thresh_grid=THRESH_GRID,
            strategy=BLEND_SELECT_STRATEGY
        )

        hist = pd.DataFrame(history)
        write_df(hist, "ensemble_tuning_trend.csv", VERIF_DIR)

        pred_nominal = (P_ens_va.argmax(1) + 1)
        acc_nominal = accuracy_score(y_va, pred_nominal)
        f1_nominal  = f1_score(y_va, pred_nominal, average="macro")
        acc_thresh  = accuracy_score(y_va, pred_best)

        metrics_ens = {
            "best_w": w_best.tolist(),
            "best_thresholds": th_best.tolist(),
            "val_acc": float(acc_nominal),
            "val_f1": float(f1_nominal),
            "val_acc_thresh": float(acc_thresh),
            "val_f1_thresh": float(f1_best)
        }
        metrics_ens = enrich_discriminative_metrics(metrics_ens, y_va, P_ens_va, pred_best)
        save_json(metrics_ens, "metrics_ens.json", VERIF_DIR)

        df_va_ens = pd.DataFrame({
            "company_id": val_ids,
            "y_true": y_va,
            "y_pred_nominal": pred_nominal,
            "y_pred_thresh": pred_best,
            "prob_z1": P_ens_va[:, 0], "prob_z2": P_ens_va[:, 1], "prob_z3": P_ens_va[:, 2]
        })
        write_df(df_va_ens, "preds_val_ens.csv", VERIF_DIR)

        _export_cm_and_report(y_va, pred_best, "ens")

        # 2) 计算 train + val 的 Ensemble 概率（用于 surrogate）
        # 训练集概率从训练输出取
        prob_tr = df_tr_lgb[["prob_z1", "prob_z2", "prob_z3"]].values.astype(float)
        prob_tr_b = df_tr_xgb[["prob_z1", "prob_z2", "prob_z3"]].values.astype(float)

        P_ens_tr = w_best[None, :] * prob_tr + (1.0 - w_best[None, :]) * prob_tr_b
        P_ens_all = np.vstack([P_ens_tr, P_ens_va])

        train_ids_str = [str(c) for c in train_ids]
        val_ids_str = [str(c) for c in val_ids]
        companies_all = train_ids_str + val_ids_str

        z_star_ens = P_ens_all.argmax(axis=1) + 1  # Ensemble 的最终类别

        # 3) Surrogate multinomial LR: 拟合 Ensemble 的决策
        X_all_lin = X_lin.reindex(companies_all).fillna(0.0)
        y_hat = z_star_ens.astype(int)  # 1/2/3

        class_counts = pd.Series(y_hat).value_counts().to_dict()
        total = len(y_hat)
        cls_w = {int(c): total / (len(class_counts) * class_counts[c]) for c in class_counts}

        lr = LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            class_weight=cls_w
        )
        lr.fit(X_all_lin.values, y_hat)

        prob_lr_raw = lr.predict_proba(X_all_lin.values)
        prob_all_lr = np.zeros((len(y_hat), 3), dtype=float)
        for j, cls in enumerate(lr.classes_):
            cls = int(cls)
            if 1 <= cls <= 3:
                prob_all_lr[:, cls - 1] = prob_lr_raw[:, j]

        pred_all_lr = lr.predict(X_all_lin.values).astype(int)

        rows = []
        feat_names = X_all_lin.columns.tolist()
        for k, cls in enumerate(lr.classes_):
            cls = int(cls)
            coefs = lr.coef_[k]
            for feat, c in zip(feat_names, coefs):
                rows.append({"z_label": cls, "feature": str(feat), "coef": float(c)})
        coef_df = pd.DataFrame(rows)
        write_df(coef_df, "coef_multinomial_long.csv", OUT_DIR)

        y_map_true = y_df.set_index("company_id")["y_true"].astype(int)
        y_true_all = y_map_true.reindex(companies_all).values

        preds_multi = pd.DataFrame({
            "company_id": companies_all,
            "y_true": y_true_all,
            "y_pred": z_star_ens,
            "prob_z1": prob_all_lr[:, 0],
            "prob_z2": prob_all_lr[:, 1],
            "prob_z3": prob_all_lr[:, 2],
            "z_star": z_star_ens
        })
        write_df(preds_multi, "preds_multinomial.csv", OUT_DIR)
        preds_multinomial_written = True

        acc_match = accuracy_score(z_star_ens, pred_all_lr)
        save_json({"surrogate_vs_ensemble_acc": float(acc_match)}, "metrics_logit_surrogate.json", VERIF_DIR)

        self_check_preds(preds_multi, input_report)

    # 汇总
    if metrics_ens is not None:
        metrics_summary["ENSEMBLE"] = metrics_ens
    save_json(metrics_summary, "metrics_summary.json", VERIF_DIR)

    # 导出对比表（不画图）
    _export_model_compare_tables(metrics_summary.get("LGBM"), metrics_summary.get("XGB"), metrics_ens)

    # 兜底：若没有 Ensemble，但仍希望 attribution 可跑
    if (not preds_multinomial_written) and ("LGBM" in metrics_summary):
        print("[WARN] Ensemble 未生成，回退输出 LGBM preds_multinomial.csv（仅为保证流水线可跑）")
        preds_all_fb = pd.concat([df_tr_lgb.assign(split="train"), df_va_lgb.assign(split="val")], axis=0, ignore_index=True)
        # 用 nominal 概率的 argmax 做 z_star
        probs = preds_all_fb[["prob_z1", "prob_z2", "prob_z3"]].values
        preds_all_fb["z_star"] = probs.argmax(axis=1) + 1
        preds_all_fb = preds_all_fb.rename(columns={"y_pred_nominal": "y_pred"})
        write_df(preds_all_fb[["company_id", "y_true", "y_pred", "prob_z1", "prob_z2", "prob_z3", "z_star"]],
                 "preds_multinomial.csv", OUT_DIR)
        self_check_preds(preds_all_fb, input_report)

    print("\n[Done] ✓")
    print("  - 训练与导出完成（不画图）")
    print("  - predictor_display.py 将读取 output/verification/preds_val_*.csv 来生成论文图")
    return


if __name__ == "__main__":
    main()
