# -*- coding: utf-8 -*-
"""
attribution.py — per-company role×risk attribution for RQ1 (reads from predict/output root)

Outputs (in predict/output/attribution/conclusion/):
  - fig_01_toprole_share_z1.png
  - fig_02_toprole_share_z2.png
  - fig_03_toprole_share_z3.png
  - fig_04_toprisk_share_z1.png   (donut pie)
  - fig_05_toprisk_share_z2.png   (donut pie)
  - fig_06_toprisk_share_z3.png   (donut pie)
  - fig_07_pair_top1_z1.png
  - fig_08_pair_top1_z2.png
  - fig_09_pair_top1_z3.png
  - z1_pair_top_counts.csv / z2_pair_top_counts.csv / z3_pair_top_counts.csv
"""

import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------- Paths --------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))     # .../predict
OUT_DIR  = os.path.join(BASE_DIR, "output")
LOG_DIR  = os.path.join(OUT_DIR, "logs")
VERIF_DIR= os.path.join(OUT_DIR, "verification")

ATT_DIR  = os.path.join(OUT_DIR, "attribution")
CONC_DIR = os.path.join(ATT_DIR, "conclusion")
os.makedirs(ATT_DIR, exist_ok=True)
os.makedirs(CONC_DIR, exist_ok=True)

# default filenames
A_MIJ_NAME = "a_mij_clean.csv"
FIJZ_NAME  = "fijz_logratio.csv"
XBASE_NAME = "features_X_base.csv"        # optional
COEF_NAME  = "coef_multinomial_long.csv"
PREDS_NAME = "preds_multinomial.csv"

CANON_RISKS = ["risk_uncertainty", "risk_legal", "risk_constraint", "risk_external"]
RISK_PRETTY = {
    "risk_uncertainty": "Uncertainty",
    "risk_legal": "Legal",
    "risk_constraint": "Constraint",
    "risk_external": "External",
}

# Low-saturation, journal-friendly palette for risk markers
RISK_COLORS = {
    "risk_uncertainty": "#4C566A",  # muted gray-blue
    "risk_legal":       "#5E81AC",  # muted blue
    "risk_constraint":  "#A3BE8C",  # muted green
    "risk_external":    "#D08770",  # muted orange
}


# =============================================================================
# Plot style (academic, minimal)
# =============================================================================
def set_academic_mpl_style():
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.9,
        "lines.linewidth": 2.0,
    })


# -------- Utils --------
def _write_df(df: pd.DataFrame, name: str, folder: str = ATT_DIR):
    os.makedirs(folder, exist_ok=True)
    p = os.path.join(folder, name)
    df.to_csv(p, index=False)
    print(f"[WRITE] {p} rows={len(df)}")
    return p

def _save_fig(fig, name: str, folder: str = ATT_DIR):
    os.makedirs(folder, exist_ok=True)
    p = os.path.join(folder, name)
    fig.savefig(p, dpi=320, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[PLOT] {p}")
    return p

def _resolve_path(fname: str, search_dirs):
    for d in search_dirs:
        p = os.path.join(d, fname)
        if os.path.exists(p):
            return p
    return None

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


# -------- Load --------
def load_inputs():
    search_dirs = [OUT_DIR, LOG_DIR, VERIF_DIR]

    a_path    = _resolve_path(A_MIJ_NAME, search_dirs)
    fijz_path = _resolve_path(FIJZ_NAME,  search_dirs)
    coef_path = _resolve_path(COEF_NAME,  search_dirs)
    preds_path= _resolve_path(PREDS_NAME, search_dirs)
    xbase_path= _resolve_path(XBASE_NAME, search_dirs)  # optional

    missing = []
    if a_path is None:     missing.append(A_MIJ_NAME)
    if fijz_path is None:  missing.append(FIJZ_NAME)
    if coef_path is None:  missing.append(COEF_NAME)
    if preds_path is None: missing.append(PREDS_NAME)

    if missing:
        raise FileNotFoundError(
            f"Missing inputs in output/ (or logs/verification fallback): {missing}. "
            f"Please run risk_predictor.py first."
        )

    print("[LOAD]")
    print("  a_mij  :", a_path)
    print("  fijz   :", fijz_path)
    print("  coef   :", coef_path)
    print("  preds  :", preds_path)
    if xbase_path:
        print("  x_base :", xbase_path)

    a = pd.read_csv(a_path)
    f = pd.read_csv(fijz_path)
    coef = pd.read_csv(coef_path)
    preds = pd.read_csv(preds_path)

    # a_mij
    for c in ["company_id", "role", "risk_bucket", "a_mij"]:
        if c not in a.columns:
            raise ValueError(f"[a_mij_clean.csv] missing column: {c}. Got: {list(a.columns)}")
    a["company_id"]  = a["company_id"].astype(str)
    a["role"]        = a["role"].astype(str).str.strip()
    a["risk_bucket"] = a["risk_bucket"].astype(str).str.strip()
    a["a_mij"]       = pd.to_numeric(a["a_mij"], errors="coerce").fillna(0.0)
    a = a[a["risk_bucket"].isin(CANON_RISKS)].copy()

    # fijz
    for c in ["role", "risk_bucket", "z_label", "f_ijz"]:
        if c not in f.columns:
            raise ValueError(f"[fijz_logratio.csv] missing column: {c}. Got: {list(f.columns)}")
    f["role"]        = f["role"].astype(str).str.strip()
    f["risk_bucket"] = f["risk_bucket"].astype(str).str.strip()
    f["z_label"]     = pd.to_numeric(f["z_label"], errors="coerce").astype("Int64")
    f["f_ijz"]       = pd.to_numeric(f["f_ijz"], errors="coerce").fillna(0.0)
    f = f[f["risk_bucket"].isin(CANON_RISKS) & f["z_label"].notna()].copy()
    f["z_label"] = f["z_label"].astype(int)

    # coef
    coef_cols = {c.lower(): c for c in coef.columns}
    c_z   = coef_cols.get("z_label") or coef_cols.get("class") or coef_cols.get("z")
    c_f   = coef_cols.get("feature") or coef_cols.get("feat")
    c_b   = coef_cols.get("coef") or coef_cols.get("beta") or coef_cols.get("weight")
    if not (c_z and c_f and c_b):
        raise ValueError(f"[coef_multinomial_long.csv] need z_label/feature/coef cols. Got: {list(coef.columns)}")

    coef = coef[[c_z, c_f, c_b]].rename(columns={c_z: "z_label", c_f: "feature", c_b: "coef"})
    coef["z_label"] = pd.to_numeric(coef["z_label"], errors="coerce").astype("Int64")
    coef["feature"] = coef["feature"].astype(str)
    coef["coef"]    = pd.to_numeric(coef["coef"], errors="coerce").fillna(0.0)
    coef = coef[coef["z_label"].notna()].copy()
    coef["z_label"] = coef["z_label"].astype(int)

    # preds
    preds["company_id"] = preds["company_id"].astype(str)
    prob_cols = _infer_prob_cols(preds)
    if prob_cols is None:
        raise ValueError(
            f"[preds_multinomial.csv] cannot find prob cols like prob_z1/prob_z2/prob_z3. "
            f"Got columns: {list(preds.columns)}"
        )
    preds = preds.rename(columns={prob_cols[0]: "prob_z1", prob_cols[1]: "prob_z2", prob_cols[2]: "prob_z3"})
    for c in ["prob_z1", "prob_z2", "prob_z3"]:
        preds[c] = pd.to_numeric(preds[c], errors="coerce").fillna(0.0)

    if "z_star" not in preds.columns:
        preds["z_star"] = preds[["prob_z1", "prob_z2", "prob_z3"]].values.argmax(axis=1) + 1

    return a, f, coef, preds


# -------- Coef maps --------
def build_coef_maps(coef_long: pd.DataFrame):
    beta_map, gamma_map = {}, {}
    for z, g in coef_long.groupby("z_label"):
        beta, gamma = {}, {}
        for _, r in g.iterrows():
            feat = str(r["feature"]).strip()
            val  = float(r["coef"])
            if feat.lower() in ["intercept", "(intercept)", "bias", "const"]:
                continue
            if feat.startswith("prior_z"):
                gamma[feat] = val
            else:
                beta[feat] = val
        beta_map[int(z)]  = beta
        gamma_map[int(z)] = gamma
    return beta_map, gamma_map


# -------- Core attribution in logit space --------
def compute_logit_contrib(a_mij: pd.DataFrame, fijz: pd.DataFrame, beta_map, gamma_map):
    f_pvt = fijz.pivot_table(index=["role", "risk_bucket"],
                             columns="z_label", values="f_ijz",
                             aggfunc="first").fillna(0.0)
    f_pvt = f_pvt.reindex(columns=[1, 2, 3]).fillna(0.0)

    rows = []
    for (m, role, risk, aval) in a_mij[["company_id", "role", "risk_bucket", "a_mij"]].itertuples(index=False):
        feat = f"{role}|{risk}"
        for z in (1, 2, 3):
            beta = beta_map.get(z, {}).get(feat, 0.0)
            gmap = gamma_map.get(z, {})

            if (role, risk) in f_pvt.index:
                f1 = f_pvt.loc[(role, risk), 1]
                f2 = f_pvt.loc[(role, risk), 2]
                f3 = f_pvt.loc[(role, risk), 3]
                v1 = gmap.get("prior_z1", 0.0) * f1 * aval
                v2 = gmap.get("prior_z2", 0.0) * f2 * aval
                v3 = gmap.get("prior_z3", 0.0) * f3 * aval
            else:
                v1 = v2 = v3 = 0.0

            direct = beta * aval
            clogit = direct + v1 + v2 + v3

            rows.append({
                "company_id": m, "role": role, "risk_bucket": risk,
                "z_label": z, "a_mij": aval,
                "direct": direct, "via_prior_z1": v1, "via_prior_z2": v2, "via_prior_z3": v3,
                "C_logit": clogit
            })
    return pd.DataFrame(rows)


# -------- Prob-scale attribution --------
def to_prob_contrib(C_logit_long: pd.DataFrame, preds: pd.DataFrame):
    ref = (C_logit_long.groupby(["role", "risk_bucket", "z_label"])["C_logit"]
           .mean().rename("C_logit_ref").reset_index())
    m = C_logit_long.merge(ref, on=["role", "risk_bucket", "z_label"], how="left")
    m["C_logit_delta"] = m["C_logit"] - m["C_logit_ref"]

    probs = preds[["company_id", "prob_z1", "prob_z2", "prob_z3"]].copy()
    probs = probs.melt(id_vars="company_id", var_name="pcol", value_name="pmz")
    probs["z_label"] = probs["pcol"].str.extract(r"(\d)").astype(int)

    m = m.merge(probs[["company_id", "z_label", "pmz"]],
                on=["company_id", "z_label"], how="left")
    m["pmz"] = m["pmz"].fillna(0.0)

    m["C_prob"] = m["pmz"] * (1.0 - m["pmz"]) * m["C_logit_delta"]
    return m


# -------- Aggregations / reports --------
def export_topk(m_prob: pd.DataFrame, preds: pd.DataFrame, k=10):
    pstar = preds[["company_id", "z_star"]]
    m = m_prob.merge(pstar, on="company_id", how="left")
    g = (m[m["z_label"] == m["z_star"]].assign(C_abs=lambda d: d["C_prob"].abs()))
    topk = (g.sort_values(["company_id", "C_abs"], ascending=[True, False])
              .groupby("company_id").head(k))
    return _write_df(topk.drop(columns=["C_abs"]), "topk_per_company.csv")

def export_aggregations(C_logit_long: pd.DataFrame, C_prob_long: pd.DataFrame):
    _write_df(C_logit_long.groupby(["company_id", "role", "z_label"])["C_logit"]
              .sum().reset_index(), "agg_role_logit.csv")
    _write_df(C_prob_long.groupby(["company_id", "role", "z_label"])["C_prob"]
              .sum().reset_index(), "agg_role_prob.csv")

    _write_df(C_logit_long.groupby(["company_id", "risk_bucket", "z_label"])["C_logit"]
              .sum().reset_index(), "agg_risk_logit.csv")
    _write_df(C_prob_long.groupby(["company_id", "risk_bucket", "z_label"])["C_prob"]
              .sum().reset_index(), "agg_risk_prob.csv")


def plot_examples(topk_csv_path: str, num_companies: int = 6):
    df = pd.read_csv(topk_csv_path)
    for m, g in df.groupby("company_id"):
        z = int(g["z_label"].iloc[0])
        lab = (g.assign(key=lambda d: d["role"] + "|" + d["risk_bucket"])
                 .sort_values("C_prob", ascending=True)
                 .tail(10))

        fig, ax = plt.subplots(figsize=(9.4, 5.2))
        y = np.arange(len(lab))
        ax.hlines(y=y, xmin=0, xmax=lab["C_prob"].values, color="0.68", linewidth=1.2)
        ax.scatter(lab["C_prob"].values, y, s=40, color="0.18")
        ax.set_yticks(y)
        ax.set_yticklabels(lab["key"].tolist())
        ax.axvline(0.0, color="0.38", linewidth=1.0)
        ax.set_title(f"Top contributions (prob-scale) — company={m}, z*={z}", pad=10)
        ax.set_xlabel("Contribution to p(z*)")
        ax.grid(True, axis="x", alpha=0.18)
        ax.grid(False, axis="y")
        _save_fig(fig, f"top_contrib_company_{m}.png")

        num_companies -= 1
        if num_companies <= 0:
            break


# =============================================================================
# ==== Conclusion figures & statistics =========================================
# =============================================================================

def _safe_pos_share(x: pd.Series, eps: float = 1e-9):
    x = x.fillna(0.0)
    if (x > 0).sum() > 0:
        s = x.clip(lower=0.0)
        den = float(s.sum())
        if den <= eps:
            den = eps
        return s / den
    s = x.abs()
    den = float(s.sum())
    if den <= eps:
        den = eps
    return s / den

def _company_level_max_winner(df_z: pd.DataFrame, key_col: str):
    g = (df_z.groupby(["company_id", key_col], as_index=False)["C_prob"].sum())
    idx = g.groupby("company_id")["C_prob"].idxmax()
    winners = g.loc[idx, ["company_id", key_col]].copy()
    winners.rename(columns={key_col: "winner"}, inplace=True)
    return winners

def _wilson_ci(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return 0.0, 0.0, 0.0
    p = k / float(n)
    denom = 1.0 + (z*z)/n
    center = (p + (z*z)/(2*n)) / denom
    half = (z / denom) * np.sqrt((p*(1-p) + (z*z)/(4*n)) / n)
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return center, lo, hi


# -------------------- Fig 01–03: role dotplot with CI --------------------
def _plot_share_dotplot(counts: pd.Series, total: int, title: str, fname: str, topk: int = 12):
    total = int(max(total, 1))
    s = counts.sort_values(ascending=False).copy()
    if topk is not None and len(s) > int(topk):
        s = s.iloc[:int(topk)]

    keys = s.index.astype(str).tolist()
    ks = s.values.astype(int).tolist()

    props, lo, hi = [], [], []
    for k in ks:
        c, l, h = _wilson_ci(int(k), int(total))
        props.append(c); lo.append(l); hi.append(h)

    y = np.arange(len(keys))[::-1]

    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.hlines(y, xmin=np.array(lo), xmax=np.array(hi), color="0.62", linewidth=2.2)
    ax.scatter(props, y, s=52, color="0.15", zorder=3)

    for yi, p, k in zip(y, props, ks):
        ax.text(min(p + 0.012, 0.985), yi, f"{p:.1%}  (n={k})",
                va="center", ha="left", fontsize=10, color="0.25")

    ax.set_yticks(y)
    ax.set_yticklabels(keys)
    ax.set_xlim(0.0, min(1.0, max(0.30, max(hi) + 0.08 if hi else 0.40)))
    ax.set_xlabel("Proportion of companies (Wilson 95% CI)")
    ax.set_title(title + f"  (N={total})", pad=10)
    ax.grid(True, axis="x", alpha=0.18)
    ax.grid(False, axis="y")

    _save_fig(fig, fname, folder=CONC_DIR)


# -------------------- Fig 04–06: risk donut pie (journal-friendly) --------------------
def _plot_risk_pie_donut(counts: pd.Series, total: int, title: str, fname: str):
    """
    Fig 04–06: Donut pie for Top-1 risk-winner shares
    - muted palette
    - donut ring
    - external callouts with label, percent, n
    """
    total = int(max(total, 1))
    c = counts.reindex(CANON_RISKS).fillna(0).astype(int)
    shares = (c.values / float(total)).astype(float)
    shares = np.clip(shares, 0.0, 1.0)

    labels = [RISK_PRETTY.get(k, k) for k in CANON_RISKS]
    colors = [RISK_COLORS[k] for k in CANON_RISKS]

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    ax.set_aspect("equal")

    wedges, _, autotexts = ax.pie(
        shares,
        colors=colors,
        startangle=90,
        counterclock=False,
        labels=None,
        autopct=lambda pct: f"{pct:.0f}%" if pct >= 1 else "",
        pctdistance=0.72,
        wedgeprops=dict(width=0.40, edgecolor="white", linewidth=1.2),
    )

    for t in autotexts:
        t.set_color("0.15")
        t.set_fontsize(10)

    ax.text(0, 0, f"N={total}", ha="center", va="center", fontsize=11, color="0.25")

    for w, lab, sh, cnt in zip(wedges, labels, shares, c.values):
        ang = (w.theta2 + w.theta1) / 2.0
        ang_rad = np.deg2rad(ang)
        x = np.cos(ang_rad)
        y = np.sin(ang_rad)

        xy = (x * 0.98, y * 0.98)
        xt = 1.28 * np.sign(x)
        yt = 1.10 * y
        ha = "left" if x >= 0 else "right"

        ax.annotate(
            f"{lab}  {sh:.0%} (n={int(cnt)})",
            xy=xy,
            xytext=(xt, yt),
            ha=ha,
            va="center",
            fontsize=10,
            color="0.20",
            arrowprops=dict(
                arrowstyle="-",
                color="0.55",
                lw=1.0,
                shrinkA=0,
                shrinkB=0,
                connectionstyle="arc3,rad=0.15",
            ),
        )

    ax.set_title(title + f"  (N={total})", pad=12)

    fig.text(
        0.5, 0.03,
        "Sectors show the distribution of Top-1 risk-winner categories across firms.",
        ha="center", va="center", fontsize=9, color="0.35"
    )

    _save_fig(fig, fname, folder=CONC_DIR)


def make_population_figures(C_prob_long: pd.DataFrame, preds: pd.DataFrame):
    """
    Fig 01–03: role winners (dotplot + CI)
    Fig 04–06: risk winners (donut pie)
    """
    zstar = preds[["company_id", "z_star"]].copy()
    merged = C_prob_long.merge(zstar, on="company_id", how="left")

    for z in (1, 2, 3):
        df_z = merged[merged["z_label"] == z].copy()
        base = int(df_z["company_id"].nunique())
        if base <= 0:
            continue

        winners_role = _company_level_max_winner(df_z, "role")
        counts_role = winners_role["winner"].value_counts()
        _plot_share_dotplot(
            counts_role, base,
            title=f"Top role share among companies (z={z})",
            fname=f"fig_{z:02d}_toprole_share_z{z}.png",
            topk=12
        )

        winners_risk = _company_level_max_winner(df_z, "risk_bucket")
        counts_risk = winners_risk["winner"].value_counts()
        _plot_risk_pie_donut(
            counts_risk, base,
            title=f"Top risk-cause share among companies (z={z})",
            fname=f"fig_{z+3:02d}_toprisk_share_z{z}.png"
        )


# =============================================================================
# ==== Population-wide role×risk winners (fig 7–9) ============================
# =============================================================================

def _pair_share_per_company(df_z: pd.DataFrame, use_amij: bool = False) -> pd.DataFrame:
    val_col = "a_mij" if use_amij else "C_prob"
    df = df_z.copy()
    df["pair"] = df["role"] + "|" + df["risk_bucket"]

    shares = []
    for m, g in df.groupby("company_id"):
        s = _safe_pos_share(g.set_index("pair")[val_col])
        tmp = s.rename("share").reset_index()
        tmp.insert(0, "company_id", m)
        shares.append(tmp)
    return pd.concat(shares, ignore_index=True) if shares else pd.DataFrame(columns=["company_id", "pair", "share"])


def _plot_pair_top1_bubble_matrix(pair_share: pd.DataFrame, base_n: int, z: int, max_roles: int = 12):
    if pair_share.empty or base_n <= 0:
        return

    idx = pair_share.groupby("company_id")["share"].idxmax()
    winners = pair_share.loc[idx, ["company_id", "pair"]].copy()
    counts = winners["pair"].value_counts()

    out = (counts.rename("count").to_frame()
           .assign(share=lambda d: d["count"] / float(base_n))
           .reset_index().rename(columns={"index": "pair"}))
    _write_df(out, f"z{z}_pair_top_counts.csv", folder=CONC_DIR)

    tmp = out.copy()
    tmp["role"] = tmp["pair"].str.split("|", n=1, expand=True)[0]
    tmp["risk_bucket"] = tmp["pair"].str.split("|", n=1, expand=True)[1]

    mat = tmp.pivot_table(index="role", columns="risk_bucket", values="share", aggfunc="sum").fillna(0.0)
    for c_ in CANON_RISKS:
        if c_ not in mat.columns:
            mat[c_] = 0.0
    mat = mat[CANON_RISKS]

    role_order = mat.sum(axis=1).sort_values(ascending=False)
    if len(role_order) > max_roles:
        role_order = role_order.iloc[:max_roles]
    mat = mat.loc[role_order.index]

    roles = mat.index.tolist()
    risks = mat.columns.tolist()
    X, Y = np.meshgrid(np.arange(len(risks)), np.arange(len(roles)))
    vals = mat.values

    vmax = float(np.max(vals)) if vals.size else 0.0
    vmax = max(vmax, 1e-9)

    min_s = 40.0
    max_s = 1300.0
    svals = min_s + (np.sqrt(vals / vmax) * (max_s - min_s))

    fig_h = max(5.6, 0.42 * len(roles) + 2.2)
    fig, ax = plt.subplots(figsize=(10.2, fig_h))

    ax.set_axisbelow(True)
    ax.grid(which="major", color="0.92", linewidth=1.0)

    sc = ax.scatter(
        X.flatten(), Y.flatten(),
        s=svals.flatten(),
        c=vals.flatten(),
        cmap="cividis",
        vmin=0.0, vmax=vmax,
        edgecolor="white",
        linewidth=0.9,
        alpha=0.95
    )

    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            v = float(vals[i, j])
            if v >= max(0.03, 0.18 * vmax):
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9, color="white")

    ax.set_xticks(np.arange(len(risks)))
    ax.set_xticklabels([RISK_PRETTY.get(r, r) for r in risks], rotation=0)
    ax.set_yticks(np.arange(len(roles)))
    ax.set_yticklabels(roles)

    ax.set_title(f"Top-1 role×risk winners across companies (z={z})", pad=10)
    ax.set_xlabel("Risk bucket")
    ax.set_ylabel("Role")

    ax.set_xlim(-0.6, len(risks) - 0.4)
    ax.set_ylim(len(roles) - 0.6, -0.4)

    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Proportion of companies", rotation=90)

    _save_fig(fig, f"fig_{6+z:02d}_pair_top1_z{z}.png", folder=CONC_DIR)


def make_population_pair_topk(C_prob_long: pd.DataFrame, preds: pd.DataFrame, use_amij: bool = False):
    merged = C_prob_long.copy()
    for z in (1, 2, 3):
        df_z = merged[merged["z_label"] == z].copy()
        base = int(df_z["company_id"].nunique())
        if base <= 0:
            continue
        pair_share = _pair_share_per_company(df_z, use_amij=use_amij)
        _plot_pair_top1_bubble_matrix(pair_share, base_n=base, z=z, max_roles=12)


# =============================================================================
def main():
    set_academic_mpl_style()

    print("[Paths]")
    print("  OUT_DIR :", OUT_DIR)
    print("  LOG_DIR :", LOG_DIR)
    print("  VERIF   :", VERIF_DIR)
    print("  ATT_DIR :", ATT_DIR)
    print("  CONC_DIR:", CONC_DIR)

    a, f, coef, preds = load_inputs()
    beta_map, gamma_map = build_coef_maps(coef)

    C_logit = compute_logit_contrib(a, f, beta_map, gamma_map)
    _write_df(C_logit, "contrib_logit_long.csv")

    C_prob = to_prob_contrib(C_logit, preds)
    _write_df(C_prob, "contrib_prob_long.csv")

    topk_path = export_topk(C_prob, preds, k=10)
    export_aggregations(C_logit, C_prob)
    plot_examples(topk_path, num_companies=6)

    make_population_figures(C_prob, preds)
    make_population_pair_topk(C_prob, preds, use_amij=False)

    print("\n[Done] ✓ attribution tables & plots are in:")
    print("  -", ATT_DIR)
    print("  -", CONC_DIR)

if __name__ == "__main__":
    main()
