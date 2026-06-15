#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
d3_psychometrics_analysis.py

Analysis script for the Dark Triad open-ended text / LLM-rating project.

Inputs
------
1. d3_llm_essay_ratings_long_FORMAL.csv
   Long-format LLM ratings with one row per essay x model x repetition.

2. personality_scores_FORMAL.csv
   Participant-level questionnaire scores from the revised scoring pipeline.

Main analyses
-------------
(1) Rating reliability:
    - Within-model ICCs across repeated calls per essay, reported as both consistency ICC(C) and absolute-agreement ICC(A).
    - Between-model ICCs across models after averaging repetitions, reported as both consistency and absolute-agreement ICCs.
    - Model-model agreement compared against each model's agreement with SD3 scores.

(2) Convergent and discriminant validity:
    - Multitrait correlation matrix between participant-level LLM scores and SD3 scores.
    - Raw Pearson correlations with 95% confidence intervals.
    - Disattenuated correlations for SD3 criterion unreliability if reliability values are supplied.

(3) Prompt specificity:
    - Broad-prompt validity vs specific-prompt validity.
    - Williams dependent-correlation test and bootstrap confidence interval for the difference.

(4) Social desirability:
    - Regression of absolute LLM-SD3 discrepancies on MCSDS.
    - Comparison of MCSDS correlations with SD3 scores vs LLM-derived scores.

This script is designed to run both locally and in Google Colab.
It uses only pandas, numpy, scipy, and matplotlib.
"""

import os
import json
import argparse
import warnings
import hashlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats


# ---------------------------------------------------------------------
# User-editable defaults
# ---------------------------------------------------------------------

DEFAULT_LLM_CSV = "d3_llm_essay_ratings_long_FORMAL.csv"
DEFAULT_SCORES_CSV = "personality_scores_FORMAL.csv"
DEFAULT_OUTPUT_DIR = "d3_analysis_outputs"

TRAITS = ["Machiavellianism", "Narcissism", "Psychopathy"]

RATING_COLS = {
    "Machiavellianism": "llm_machiavellianism",
    "Narcissism": "llm_narcissism",
    "Psychopathy": "llm_psychopathy",
}

# Fill these in if you have reliability estimates for the SD3 subscales
# from your own sample or from an external benchmark. If left as None,
# disattenuated correlations are reported as missing rather than guessed.
CRITERION_RELIABILITY = {
    "Machiavellianism": None,
    "Narcissism": None,
    "Psychopathy": None,
}

N_BOOT = 2000
RANDOM_SEED = 20260612


# ---------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def parse_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes", "y"])


def zscore(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    sd = s.std(ddof=1)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.nan, index=s.index)
    return (s - s.mean()) / sd


def safe_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def first_existing_path(candidates: List[str]) -> Optional[str]:
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def clean_model_name(x: str) -> str:
    return str(x).replace("/", "_").replace(" ", "_")


# ---------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------

def pearson_with_ci(x, y, alpha: float = 0.05) -> Dict[str, float]:
    x = pd.to_numeric(pd.Series(x), errors="coerce")
    y = pd.to_numeric(pd.Series(y), errors="coerce")
    m = x.notna() & y.notna()
    x = x[m].astype(float)
    y = y[m].astype(float)
    n = len(x)
    if n < 4 or x.nunique() < 2 or y.nunique() < 2:
        return {"n": n, "r": np.nan, "p": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    r, p = stats.pearsonr(x, y)
    r = float(np.clip(r, -0.999999, 0.999999))
    z = np.arctanh(r)
    se = 1.0 / np.sqrt(n - 3)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    lo = np.tanh(z - zcrit * se)
    hi = np.tanh(z + zcrit * se)
    return {"n": n, "r": r, "p": p, "ci_low": lo, "ci_high": hi}


def disattenuate_for_criterion(r: float, ci_low: float, ci_high: float, reliability: Optional[float]) -> Tuple[float, float, float]:
    if reliability is None or pd.isna(reliability) or reliability <= 0:
        return (np.nan, np.nan, np.nan)
    denom = np.sqrt(reliability)
    vals = np.array([r, ci_low, ci_high], dtype=float) / denom
    vals = np.clip(vals, -1, 1)
    lo, hi = sorted([vals[1], vals[2]])
    return (vals[0], lo, hi)


def corr_value(x, y) -> float:
    res = pearson_with_ci(x, y)
    return res["r"]


def dependent_corr_test_shared_variable(x, y1, y2) -> Dict[str, float]:
    """
    Williams test for comparing two dependent correlations that share x:
    r(x, y1) vs r(x, y2), accounting for r(y1, y2).

    This is the standard small-sample modification often meant in practice
    when researchers refer to Steiger/Williams tests for overlapping dependent
    correlations. Bootstrap CIs are also reported and should be emphasized with N=73.
    """
    df = pd.DataFrame({"x": x, "y1": y1, "y2": y2}).apply(pd.to_numeric, errors="coerce").dropna()
    n = len(df)
    if n < 6 or df["x"].nunique() < 2 or df["y1"].nunique() < 2 or df["y2"].nunique() < 2:
        return {"test": "Williams", "n": n, "r_xy1": np.nan, "r_xy2": np.nan, "r_y1y2": np.nan,
                "diff": np.nan, "t": np.nan, "df": np.nan, "p": np.nan}

    r12 = float(stats.pearsonr(df["x"], df["y1"])[0])
    r13 = float(stats.pearsonr(df["x"], df["y2"])[0])
    r23 = float(stats.pearsonr(df["y1"], df["y2"])[0])

    # Determinant of the 3x3 correlation matrix.
    det_r = 1 - r12**2 - r13**2 - r23**2 + 2 * r12 * r13 * r23
    # Williams (1959) denominator for overlapping dependent correlations.
    denom = 2 * det_r * ((n - 1) / (n - 3)) + (((r12 + r13) ** 2) / 4.0) * ((1 - r23) ** 3)
    numer_factor = (n - 1) * (1 + r23)

    if denom <= 0 or numer_factor <= 0:
        tval = np.nan
        pval = np.nan
    else:
        tval = (r12 - r13) * np.sqrt(numer_factor / denom)
        pval = 2 * stats.t.sf(abs(tval), df=n - 3)

    return {"test": "Williams", "n": n, "r_xy1": r12, "r_xy2": r13, "r_y1y2": r23,
            "diff": r12 - r13, "t": tval, "df": n - 3, "p": pval}


def bootstrap_corr_diff_shared(x, y1, y2, n_boot: int = N_BOOT, seed: int = RANDOM_SEED) -> Dict[str, float]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({"x": x, "y1": y1, "y2": y2}).apply(pd.to_numeric, errors="coerce").dropna()
    n = len(df)
    if n < 6:
        return {"boot_n": 0, "boot_diff_mean": np.nan, "boot_ci_low": np.nan, "boot_ci_high": np.nan}

    vals = []
    arr = df.to_numpy()
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b = pd.DataFrame(arr[idx, :], columns=["x", "y1", "y2"])
        if b["x"].nunique() < 2 or b["y1"].nunique() < 2 or b["y2"].nunique() < 2:
            continue
        r1 = stats.pearsonr(b["x"], b["y1"])[0]
        r2 = stats.pearsonr(b["x"], b["y2"])[0]
        vals.append(r1 - r2)
    if len(vals) == 0:
        return {"boot_n": 0, "boot_diff_mean": np.nan, "boot_ci_low": np.nan, "boot_ci_high": np.nan}
    vals = np.array(vals)
    return {
        "boot_n": len(vals),
        "boot_diff_mean": float(np.mean(vals)),
        "boot_ci_low": float(np.percentile(vals, 2.5)),
        "boot_ci_high": float(np.percentile(vals, 97.5)),
    }


# ---------------------------------------------------------------------
# ICC helpers
# ---------------------------------------------------------------------

def _icc_components(data: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Return ANOVA components for a two-way targets x raters matrix."""
    X = data.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    n, k = X.shape
    if n < 2 or k < 2:
        return None
    arr = X.to_numpy(dtype=float)
    grand = arr.mean()
    row_means = arr.mean(axis=1)
    col_means = arr.mean(axis=0)
    ss_rows = k * np.sum((row_means - grand) ** 2)
    ss_cols = n * np.sum((col_means - grand) ** 2)
    ss_total = np.sum((arr - grand) ** 2)
    ss_error = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1)
    ms_cols = ss_cols / (k - 1)
    ms_error = ss_error / ((n - 1) * (k - 1))
    return {
        "n_targets": n,
        "n_raters": k,
        "ms_rows": float(ms_rows),
        "ms_cols": float(ms_cols),
        "ms_error": float(ms_error),
    }


def icc_two_way_consistency(data: pd.DataFrame) -> Dict[str, float]:
    """
    Two-way consistency ICC: ICC(C,1) and ICC(C,k).

    Rows are targets; columns are raters/repetitions/models. Rater/model mean
    differences are removed, so this measures rank-order consistency rather
    than absolute calibration. Missing values are handled by complete-case
    deletion across raters.
    """
    comps = _icc_components(data)
    if comps is None:
        X = data.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
        return {"n_targets": X.shape[0], "n_raters": X.shape[1],
                "icc_c_single": np.nan, "icc_c_average": np.nan,
                "ms_rows": np.nan, "ms_cols": np.nan, "ms_error": np.nan}
    n, k = comps["n_targets"], comps["n_raters"]
    ms_rows, ms_error = comps["ms_rows"], comps["ms_error"]
    denom_single = ms_rows + (k - 1) * ms_error
    if denom_single == 0:
        icc_single = np.nan
        icc_avg = np.nan
    else:
        icc_single = (ms_rows - ms_error) / denom_single
        icc_avg = (ms_rows - ms_error) / ms_rows if ms_rows != 0 else np.nan
    out = dict(comps)
    out.update({"icc_c_single": float(icc_single), "icc_c_average": float(icc_avg)})
    return out


def icc_two_way_absolute(data: pd.DataFrame) -> Dict[str, float]:
    """
    Two-way absolute-agreement ICC: ICC(A,1) and ICC(A,k).

    This retains systematic rater/model mean differences, so it measures
    absolute calibration as well as rank ordering. Missing values are handled
    by complete-case deletion across raters.
    """
    comps = _icc_components(data)
    if comps is None:
        X = data.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
        return {"n_targets": X.shape[0], "n_raters": X.shape[1],
                "icc_a_single": np.nan, "icc_a_average": np.nan,
                "ms_rows": np.nan, "ms_cols": np.nan, "ms_error": np.nan}
    n, k = comps["n_targets"], comps["n_raters"]
    ms_rows, ms_cols, ms_error = comps["ms_rows"], comps["ms_cols"], comps["ms_error"]
    denom_single = ms_rows + (k - 1) * ms_error + (k * (ms_cols - ms_error) / n)
    denom_avg = ms_rows + ((ms_cols - ms_error) / n)
    if denom_single == 0:
        icc_single = np.nan
    else:
        icc_single = (ms_rows - ms_error) / denom_single
    if denom_avg == 0:
        icc_avg = np.nan
    else:
        icc_avg = (ms_rows - ms_error) / denom_avg
    out = dict(comps)
    out.update({"icc_a_single": float(icc_single), "icc_a_average": float(icc_avg)})
    return out


def icc_two_way_both(data: pd.DataFrame) -> Dict[str, float]:
    """Return both consistency and absolute-agreement ICCs for the same matrix."""
    c = icc_two_way_consistency(data)
    a = icc_two_way_absolute(data)
    return {
        "n_targets": c.get("n_targets"),
        "n_raters": c.get("n_raters"),
        "icc_c_single": c.get("icc_c_single"),
        "icc_c_average": c.get("icc_c_average"),
        "icc_a_single": a.get("icc_a_single"),
        "icc_a_average": a.get("icc_a_average"),
        "ms_rows": c.get("ms_rows"),
        "ms_cols": c.get("ms_cols"),
        "ms_error": c.get("ms_error"),
    }


def bootstrap_icc(data: pd.DataFrame, n_boot: int = N_BOOT, seed: int = RANDOM_SEED) -> Dict[str, float]:
    X = data.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    n, k = X.shape
    if n < 4 or k < 2:
        return {
            "boot_n": 0,
            "icc_c_single_ci_low": np.nan, "icc_c_single_ci_high": np.nan,
            "icc_c_average_ci_low": np.nan, "icc_c_average_ci_high": np.nan,
            "icc_a_single_ci_low": np.nan, "icc_a_single_ci_high": np.nan,
            "icc_a_average_ci_low": np.nan, "icc_a_average_ci_high": np.nan,
        }
    rng = np.random.default_rng(seed)
    vals = {"icc_c_single": [], "icc_c_average": [], "icc_a_single": [], "icc_a_average": []}
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        b = X.iloc[idx, :]
        res = icc_two_way_both(b)
        for key in vals:
            if not pd.isna(res.get(key)):
                vals[key].append(res[key])
    out = {"boot_n": max(len(v) for v in vals.values()) if vals else 0}
    for key, v in vals.items():
        out[f"{key}_ci_low"] = float(np.percentile(v, 2.5)) if v else np.nan
        out[f"{key}_ci_high"] = float(np.percentile(v, 97.5)) if v else np.nan
    return out


# ---------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------

def load_data(llm_csv: str, scores_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    llm = pd.read_csv(llm_csv)
    scores = pd.read_csv(scores_csv)

    required_llm = [
        "essay_id", "participant_id", "target_trait", "prompt_type",
        "model", "run", "parse_success",
    ] + list(RATING_COLS.values())
    required_scores = ["participant_id", "SDS_mean"] + TRAITS

    missing_llm = [c for c in required_llm if c not in llm.columns]
    missing_scores = [c for c in required_scores if c not in scores.columns]
    if missing_llm:
        raise ValueError("LLM file is missing columns: %s" % missing_llm)
    if missing_scores:
        raise ValueError("Scores file is missing columns: %s" % missing_scores)

    llm = safe_numeric(llm, list(RATING_COLS.values()) + ["run", "word_count", "total_rt"])
    scores = safe_numeric(scores, TRAITS + ["SDS_mean", "age", "total_rt"])

    llm["parse_success_bool"] = parse_bool_series(llm["parse_success"])
    llm["participant_id"] = llm["participant_id"].astype(str)
    scores["participant_id"] = scores["participant_id"].astype(str)

    rating_ids = set(llm["participant_id"].unique())
    score_ids = set(scores["participant_id"].unique())
    if rating_ids != score_ids:
        warnings.warn(
            "participant_id sets differ. In LLM not scores: %s; in scores not LLM: %s"
            % (sorted(rating_ids - score_ids), sorted(score_ids - rating_ids))
        )

    return llm, scores


def melt_llm_ratings(llm: pd.DataFrame, successful_only: bool = True) -> pd.DataFrame:
    dat = llm.copy()
    if successful_only:
        dat = dat[dat["parse_success_bool"]].copy()
    rows = []
    id_cols = [c for c in dat.columns if c not in RATING_COLS.values()]
    for trait, col in RATING_COLS.items():
        tmp = dat[id_cols + [col]].copy()
        tmp = tmp.rename(columns={col: "llm_rating"})
        tmp["rating_trait"] = trait
        rows.append(tmp)
    out = pd.concat(rows, ignore_index=True)
    out["llm_rating"] = pd.to_numeric(out["llm_rating"], errors="coerce")
    out = out.dropna(subset=["llm_rating"])
    return out


def data_quality_summary(llm: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.append({"metric": "participants_in_scores", "value": scores["participant_id"].nunique()})
    rows.append({"metric": "participants_in_llm", "value": llm["participant_id"].nunique()})
    rows.append({"metric": "essays_in_llm", "value": llm["essay_id"].nunique()})
    rows.append({"metric": "llm_rows_total", "value": len(llm)})
    rows.append({"metric": "llm_rows_parse_success", "value": int(llm["parse_success_bool"].sum())})
    rows.append({"metric": "llm_rows_parse_failure", "value": int((~llm["parse_success_bool"]).sum())})
    rating_cols = list(RATING_COLS.values())
    llm["n_trait_ratings_present"] = llm[rating_cols].notna().sum(axis=1)
    rows.append({"metric": "llm_rows_partial_trait_ratings_present_but_not_complete", "value": int(((~llm["parse_success_bool"]) & (llm["n_trait_ratings_present"] > 0)).sum())})
    rows.append({"metric": "llm_trait_cells_available_total", "value": int(llm[rating_cols].notna().sum().sum())})
    for model, g in llm.groupby("model"):
        rows.append({"metric": "parse_success_rate_%s" % model, "value": float(g["parse_success_bool"].mean())})
        rows.append({"metric": "rows_%s" % model, "value": len(g)})
        rows.append({"metric": "partial_failed_rows_%s" % model, "value": int(((~g["parse_success_bool"]) & (g["n_trait_ratings_present"] > 0)).sum())})
        rows.append({"metric": "trait_cells_available_%s" % model, "value": int(g[rating_cols].notna().sum().sum())})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Score aggregation
# ---------------------------------------------------------------------

def make_participant_llm_scores(melted: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Create participant-level LLM scores.

    The primary score uses target-trait ratings only:
    - Machiavellianism score = Machiavellianism ratings on Machiavellianism-target essays
    - Narcissism score       = Narcissism ratings on Narcissism-target essays
    - Psychopathy score      = Psychopathy ratings on Psychopathy-target essays
    """
    target = melted[melted["target_trait"] == melted["rating_trait"]].copy()

    by_model = (
        target.groupby(["participant_id", "model", "rating_trait"], as_index=False)["llm_rating"]
        .mean()
        .rename(columns={"llm_rating": "llm_score"})
    )
    by_model["score_type"] = "target_trait_all_prompts"

    model_average = (
        by_model.groupby(["participant_id", "rating_trait"], as_index=False)["llm_score"]
        .mean()
    )
    model_average["model"] = "MODEL_AVERAGE"
    model_average["score_type"] = "target_trait_all_prompts"

    by_model_prompt = (
        target.groupby(["participant_id", "model", "prompt_type", "rating_trait"], as_index=False)["llm_rating"]
        .mean()
        .rename(columns={"llm_rating": "llm_score"})
    )
    by_model_prompt["score_type"] = "target_trait_by_prompt"

    model_average_prompt = (
        by_model_prompt.groupby(["participant_id", "prompt_type", "rating_trait"], as_index=False)["llm_score"]
        .mean()
    )
    model_average_prompt["model"] = "MODEL_AVERAGE"
    model_average_prompt["score_type"] = "target_trait_by_prompt"

    all_scores = pd.concat([
        by_model[["participant_id", "model", "rating_trait", "llm_score", "score_type"]],
        model_average[["participant_id", "model", "rating_trait", "llm_score", "score_type"]],
        by_model_prompt[["participant_id", "model", "prompt_type", "rating_trait", "llm_score", "score_type"]],
        model_average_prompt[["participant_id", "model", "prompt_type", "rating_trait", "llm_score", "score_type"]],
    ], ignore_index=True, sort=False)

    return {
        "target_by_model": by_model,
        "target_model_average": model_average,
        "target_by_model_prompt": by_model_prompt,
        "target_model_average_prompt": model_average_prompt,
        "all_long": all_scores,
    }


def wide_scores_for_model(score_long: pd.DataFrame, model: str, score_type: str = "target_trait_all_prompts") -> pd.DataFrame:
    """
    Return participant-level wide LLM scores for one model.

    Important: score_long may contain both overall scores and by-prompt scores.
    We therefore filter to score_type='target_trait_all_prompts' by default;
    otherwise broad/specific prompt scores would be averaged together with the
    intended all-prompt score, which is wrong when missingness is unbalanced.
    """
    dat = score_long[(score_long["model"] == model) & (score_long["score_type"] == score_type)].copy()
    wide = dat.pivot_table(
        index="participant_id",
        columns="rating_trait",
        values="llm_score",
        aggfunc="mean",
    ).reset_index()
    wide.columns.name = None
    wide = wide.rename(columns={t: "LLM_%s" % t for t in TRAITS})
    return wide


# ---------------------------------------------------------------------
# Analyses
# ---------------------------------------------------------------------

def analyze_within_model_icc(melted: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for essay_set in ["all_essays", "target_trait_essays"]:
        if essay_set == "all_essays":
            dat0 = melted.copy()
        else:
            dat0 = melted[melted["target_trait"] == melted["rating_trait"]].copy()

        for (model, trait), g in dat0.groupby(["model", "rating_trait"]):
            mat = g.pivot_table(index="essay_id", columns="run", values="llm_rating", aggfunc="mean")
            res = icc_two_way_both(mat)
            ci = bootstrap_icc(mat, n_boot=n_boot, seed=seed)
            row = {"essay_set": essay_set, "model": model, "rating_trait": trait}
            row.update(res)
            row.update(ci)
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_between_model_icc(melted: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    rep_avg = (
        melted.groupby(["essay_id", "target_trait", "rating_trait", "model"], as_index=False)["llm_rating"]
        .mean()
    )
    for essay_set in ["all_essays", "target_trait_essays"]:
        if essay_set == "all_essays":
            dat0 = rep_avg.copy()
        else:
            dat0 = rep_avg[rep_avg["target_trait"] == rep_avg["rating_trait"]].copy()

        for trait, g in dat0.groupby("rating_trait"):
            mat = g.pivot_table(index="essay_id", columns="model", values="llm_rating", aggfunc="mean")
            res = icc_two_way_both(mat)
            ci = bootstrap_icc(mat, n_boot=n_boot, seed=seed)
            row = {"essay_set": essay_set, "rating_trait": trait}
            row.update(res)
            row.update(ci)
            rows.append(row)
    return pd.DataFrame(rows)


def analyze_multitrait_validity(score_long: pd.DataFrame, scores: pd.DataFrame, reliability: Dict[str, Optional[float]]) -> pd.DataFrame:
    rows = []
    for model in sorted(score_long["model"].unique()):
        wide = wide_scores_for_model(score_long, model).merge(scores[["participant_id"] + TRAITS], on="participant_id", how="inner")
        for llm_trait in TRAITS:
            for sd3_trait in TRAITS:
                x = wide["LLM_%s" % llm_trait]
                y = wide[sd3_trait]
                res = pearson_with_ci(x, y)
                dis_r, dis_lo, dis_hi = disattenuate_for_criterion(
                    res["r"], res["ci_low"], res["ci_high"], reliability.get(sd3_trait)
                )
                rows.append({
                    "model": model,
                    "llm_trait": llm_trait,
                    "sd3_trait": sd3_trait,
                    "match_type": "convergent" if llm_trait == sd3_trait else "discriminant",
                    "n": res["n"],
                    "r": res["r"],
                    "p": res["p"],
                    "ci_low": res["ci_low"],
                    "ci_high": res["ci_high"],
                    "criterion_reliability": reliability.get(sd3_trait),
                    "r_disattenuated_for_criterion": dis_r,
                    "disattenuated_ci_low": dis_lo,
                    "disattenuated_ci_high": dis_hi,
                })
    return pd.DataFrame(rows)


def analyze_matched_vs_mismatched(validity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, g in validity.groupby("model"):
        conv = g[g["match_type"] == "convergent"]["r"].dropna()
        disc = g[g["match_type"] == "discriminant"]["r"].dropna()
        rows.append({
            "model": model,
            "mean_convergent_r": conv.mean() if len(conv) else np.nan,
            "median_convergent_r": conv.median() if len(conv) else np.nan,
            "mean_discriminant_r": disc.mean() if len(disc) else np.nan,
            "median_discriminant_r": disc.median() if len(disc) else np.nan,
            "mean_convergent_minus_discriminant": (conv.mean() - disc.mean()) if len(conv) and len(disc) else np.nan,
        })
    return pd.DataFrame(rows)


def analyze_model_model_vs_sd3(score_long: pd.DataFrame, scores: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pairwise model-model agreement among participant-level target-trait LLM scores,
    and model-SD3 matched correlations.
    """
    pair_rows = []
    sd3_rows = []
    models = sorted([m for m in score_long["model"].unique() if m != "MODEL_AVERAGE"])

    # Long to wide with columns model_trait
    for trait in TRAITS:
        dat = score_long[(score_long["rating_trait"] == trait) & (score_long["model"].isin(models))]
        wide = dat.pivot_table(index="participant_id", columns="model", values="llm_score", aggfunc="mean").reset_index()
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                if m1 in wide.columns and m2 in wide.columns:
                    res = pearson_with_ci(wide[m1], wide[m2])
                    pair_rows.append({
                        "trait": trait, "model_1": m1, "model_2": m2,
                        "n": res["n"], "r": res["r"], "p": res["p"], "ci_low": res["ci_low"], "ci_high": res["ci_high"]
                    })

        wide_sd3 = wide.merge(scores[["participant_id", trait]], on="participant_id", how="inner")
        for m in models:
            if m in wide_sd3.columns:
                res = pearson_with_ci(wide_sd3[m], wide_sd3[trait])
                sd3_rows.append({
                    "trait": trait, "model": m,
                    "n": res["n"], "r_model_sd3": res["r"], "p": res["p"],
                    "ci_low": res["ci_low"], "ci_high": res["ci_high"]
                })

    return pd.DataFrame(pair_rows), pd.DataFrame(sd3_rows)


def analyze_prompt_specificity(prompt_scores: pd.DataFrame, scores: pd.DataFrame, n_boot: int, seed: int) -> pd.DataFrame:
    rows = []
    for model in sorted(prompt_scores["model"].unique()):
        for trait in TRAITS:
            dat = prompt_scores[(prompt_scores["model"] == model) & (prompt_scores["rating_trait"] == trait)]
            wide = dat.pivot_table(index="participant_id", columns="prompt_type", values="llm_score", aggfunc="mean").reset_index()
            if "broad" not in wide.columns or "specific" not in wide.columns:
                continue
            wide = wide.merge(scores[["participant_id", trait]], on="participant_id", how="inner")
            test = dependent_corr_test_shared_variable(wide[trait], wide["broad"], wide["specific"])
            boot = bootstrap_corr_diff_shared(wide[trait], wide["broad"], wide["specific"], n_boot=n_boot, seed=seed)
            rows.append({
                "model": model,
                "trait": trait,
                "n": test["n"],
                "r_sd3_broad": test["r_xy1"],
                "r_sd3_specific": test["r_xy2"],
                "r_broad_specific": test["r_y1y2"],
                "diff_broad_minus_specific": test["diff"],
                "dependent_corr_test": test.get("test", "Williams"),
                "williams_t": test["t"],
                "williams_df": test["df"],
                "williams_p": test["p"],
                "boot_diff_mean": boot["boot_diff_mean"],
                "boot_ci_low": boot["boot_ci_low"],
                "boot_ci_high": boot["boot_ci_high"],
                "boot_n": boot["boot_n"],
            })
    return pd.DataFrame(rows)


def ols_simple(y, x) -> Dict[str, float]:
    df = pd.DataFrame({"y": y, "x": x}).apply(pd.to_numeric, errors="coerce").dropna()
    n = len(df)
    if n < 4 or df["x"].nunique() < 2:
        return {"n": n, "intercept": np.nan, "slope": np.nan, "slope_se": np.nan,
                "slope_ci_low": np.nan, "slope_ci_high": np.nan, "t": np.nan, "p": np.nan, "r2": np.nan}
    X = np.column_stack([np.ones(n), df["x"].to_numpy()])
    yv = df["y"].to_numpy()
    beta = np.linalg.inv(X.T @ X) @ (X.T @ yv)
    resid = yv - X @ beta
    df_resid = n - 2
    mse = np.sum(resid ** 2) / df_resid
    cov = mse * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tval = beta[1] / se[1] if se[1] != 0 else np.nan
    pval = 2 * stats.t.sf(abs(tval), df=df_resid) if not pd.isna(tval) else np.nan
    ci = beta[1] + np.array([-1, 1]) * stats.t.ppf(0.975, df=df_resid) * se[1]
    ss_total = np.sum((yv - yv.mean()) ** 2)
    r2 = 1 - np.sum(resid ** 2) / ss_total if ss_total != 0 else np.nan
    return {
        "n": n, "intercept": beta[0], "slope": beta[1], "slope_se": se[1],
        "slope_ci_low": ci[0], "slope_ci_high": ci[1],
        "t": tval, "p": pval, "r2": r2
    }


def analyze_social_desirability(score_long: pd.DataFrame, scores: pd.DataFrame, n_boot: int, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    discrepancy_rows = []
    corr_compare_rows = []

    for model in sorted(score_long["model"].unique()):
        wide = wide_scores_for_model(score_long, model).merge(
            scores[["participant_id", "SDS_mean"] + TRAITS], on="participant_id", how="inner"
        )
        wide["MCSDS_z"] = zscore(wide["SDS_mean"])

        for trait in TRAITS:
            llm_col = "LLM_%s" % trait
            # Use z-standardized absolute discrepancy to avoid differences in scale use.
            wide["llm_z"] = zscore(wide[llm_col])
            wide["sd3_z"] = zscore(wide[trait])
            discrepancy = (wide["llm_z"] - wide["sd3_z"]).abs()
            reg = ols_simple(discrepancy, wide["MCSDS_z"])
            row = {"model": model, "trait": trait, "outcome": "abs_z_LLM_minus_SD3", "predictor": "MCSDS_z"}
            row.update(reg)
            discrepancy_rows.append(row)

            # Compare r(MCSDS, SD3_trait) vs r(MCSDS, LLM_trait)
            test = dependent_corr_test_shared_variable(wide["SDS_mean"], wide[trait], wide[llm_col])
            boot = bootstrap_corr_diff_shared(wide["SDS_mean"], wide[trait], wide[llm_col], n_boot=n_boot, seed=seed)
            corr_compare_rows.append({
                "model": model,
                "trait": trait,
                "n": test["n"],
                "r_mcsds_sd3": test["r_xy1"],
                "r_mcsds_llm": test["r_xy2"],
                "r_sd3_llm": test["r_y1y2"],
                "diff_mcsds_sd3_minus_mcsds_llm": test["diff"],
                "dependent_corr_test": test.get("test", "Williams"),
                "williams_t": test["t"],
                "williams_df": test["df"],
                "williams_p": test["p"],
                "boot_diff_mean": boot["boot_diff_mean"],
                "boot_ci_low": boot["boot_ci_low"],
                "boot_ci_high": boot["boot_ci_high"],
                "boot_n": boot["boot_n"],
            })

    return pd.DataFrame(discrepancy_rows), pd.DataFrame(corr_compare_rows)


def analyze_questionnaire_benchmarks(scores: pd.DataFrame) -> pd.DataFrame:
    vars_ = TRAITS + [c for c in ["SDS_mean", "Extraversion", "Agreeableness", "Conscientiousness", "Neuroticism", "Intellect"] if c in scores.columns]
    rows = []
    for i in range(len(vars_)):
        for j in range(i + 1, len(vars_)):
            a, b = vars_[i], vars_[j]
            res = pearson_with_ci(scores[a], scores[b])
            rows.append({
                "var_1": a, "var_2": b,
                "n": res["n"], "r": res["r"], "p": res["p"], "ci_low": res["ci_low"], "ci_high": res["ci_high"]
            })
    return pd.DataFrame(rows)



# ---------------------------------------------------------------------
# Optional SD3 criterion reliability from item-level long data
# ---------------------------------------------------------------------

LIKERT_MAP = {
    "Disagree Strongly": 1,
    "Disagree": 2,
    "Neither Agree nor Disagree": 3,
    "Agree": 4,
    "Agree Strongly": 5,
}

SD3_ITEMS = {
    "Machiavellianism": ["sd3m%d" % i for i in range(1, 10)],
    "Narcissism": ["sd3n%d" % i for i in range(1, 10)],
    "Psychopathy": ["sd3p%d" % i for i in range(1, 10)],
}
SD3_REVERSE_ITEMS = set(["sd3n2", "sd3n6", "sd3n8", "sd3p2", "sd3p7"])


def map_likert_value(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if text in LIKERT_MAP:
        return float(LIKERT_MAP[text])
    try:
        return float(text)
    except ValueError:
        return np.nan


def cronbach_alpha(item_df: pd.DataFrame) -> float:
    X = item_df.apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    n, k = X.shape
    if n < 3 or k < 2:
        return np.nan
    item_vars = X.var(axis=0, ddof=1)
    total_var = X.sum(axis=1).var(ddof=1)
    if pd.isna(total_var) or total_var == 0:
        return np.nan
    return float((k / (k - 1)) * (1 - item_vars.sum() / total_var))


def make_participant_id_from_row(row) -> str:
    raw = "|".join([
        str(row.get("run_id", "")),
        str(row.get("recorded_at", "")),
        str(row.get("ip", "")),
        str(row.get("user_agent", "")),
    ])
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def compute_sd3_reliabilities_from_long(item_long_csv: str, participant_ids: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compute Cronbach's alpha for each SD3 subscale from a stats_ready_long.csv-style file.
    This is optional. If supplied, the resulting alphas can be used as criterion
    reliabilities for disattenuating LLM-SD3 correlations.
    """
    dat = pd.read_csv(item_long_csv)
    if "participant_id" not in dat.columns:
        hash_cols = {"run_id", "recorded_at", "ip", "user_agent"}
        if hash_cols.issubset(set(dat.columns)):
            dat["participant_id"] = dat.apply(make_participant_id_from_row, axis=1)
        else:
            raise ValueError(
                "Item-level long file needs participant_id, or the columns needed to reconstruct it: %s"
                % sorted(hash_cols)
            )
    required = {"participant_id", "question_id", "value"}
    missing = required - set(dat.columns)
    if missing:
        raise ValueError("Item-level long file missing columns required for alpha: %s" % sorted(missing))
    dat["participant_id"] = dat["participant_id"].astype(str)
    if participant_ids is not None:
        keep = set(str(x) for x in participant_ids)
        dat = dat[dat["participant_id"].isin(keep)].copy()
    needed_items = set(sum(SD3_ITEMS.values(), []))
    dat = dat[dat["question_id"].astype(str).isin(needed_items)].copy()
    wide = dat.pivot_table(index="participant_id", columns="question_id", values="value", aggfunc=lambda x: x.dropna().iloc[-1] if len(x.dropna()) else pd.NA)
    rows = []
    for trait, items in SD3_ITEMS.items():
        for item in items:
            if item not in wide.columns:
                wide[item] = pd.NA
        num = wide[items].apply(lambda col: col.map(map_likert_value))
        for item in items:
            if item in SD3_REVERSE_ITEMS:
                num[item] = num[item].apply(lambda x: 6 - x if pd.notna(x) else x)
        alpha = cronbach_alpha(num)
        complete_n = int(num.dropna(axis=0, how="any").shape[0])
        rows.append({
            "trait": trait,
            "n_complete_item_cases": complete_n,
            "n_items": len(items),
            "cronbach_alpha": alpha,
        })
    return pd.DataFrame(rows)


def merge_reliability_defaults(criterion_reliability: Dict[str, Optional[float]], alpha_df: Optional[pd.DataFrame]) -> Dict[str, Optional[float]]:
    rel = dict(criterion_reliability)
    if alpha_df is None or alpha_df.empty:
        return rel
    for _, row in alpha_df.iterrows():
        trait = row.get("trait")
        alpha = row.get("cronbach_alpha")
        if trait in rel and (rel[trait] is None or pd.isna(rel[trait])) and pd.notna(alpha):
            rel[trait] = float(alpha)
    return rel


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def run_analysis(
    llm_csv: str,
    scores_csv: str,
    output_dir: str,
    criterion_reliability: Dict[str, Optional[float]],
    n_boot: int = N_BOOT,
    seed: int = RANDOM_SEED,
    item_long_csv: Optional[str] = None,
) -> None:
    ensure_dir(output_dir)
    print("Loading data...")
    llm, scores = load_data(llm_csv, scores_csv)
    print("LLM rows: %d; score rows: %d" % (len(llm), len(scores)))

    quality = data_quality_summary(llm, scores)
    quality.to_csv(os.path.join(output_dir, "00_data_quality_summary.csv"), index=False)
    print(quality.to_string(index=False))

    alpha_df = None
    if item_long_csv is not None and str(item_long_csv).strip() != "":
        print("Computing SD3 criterion reliabilities from item-level long file...")
        alpha_df = compute_sd3_reliabilities_from_long(item_long_csv, participant_ids=list(scores["participant_id"].astype(str)))
        alpha_df.to_csv(os.path.join(output_dir, "16_sd3_reliability_alphas.csv"), index=False)
        criterion_reliability = merge_reliability_defaults(criterion_reliability, alpha_df)
        print(alpha_df.to_string(index=False))
        print("Criterion reliabilities used:", criterion_reliability)

    melted = melt_llm_ratings(llm, successful_only=True)
    print("Primary complete-parse melted rating rows:", len(melted))

    print("Aggregating participant-level LLM scores...")
    score_tables = make_participant_llm_scores(melted)
    score_tables["all_long"].to_csv(os.path.join(output_dir, "01_participant_llm_scores_long.csv"), index=False)

    # Also save a convenient wide model-average participant file.
    model_avg_wide = wide_scores_for_model(score_tables["target_model_average"], "MODEL_AVERAGE")
    participant_joined = model_avg_wide.merge(scores, on="participant_id", how="inner")
    participant_joined.to_csv(os.path.join(output_dir, "01_participant_model_average_scores_wide.csv"), index=False)

    print("Computing ICC reliability...")
    within_icc = analyze_within_model_icc(melted, n_boot=n_boot, seed=seed)
    within_icc.to_csv(os.path.join(output_dir, "02_within_model_repetition_icc.csv"), index=False)

    between_icc = analyze_between_model_icc(melted, n_boot=n_boot, seed=seed)
    between_icc.to_csv(os.path.join(output_dir, "03_between_model_icc.csv"), index=False)

    print("Computing model-model and model-SD3 agreement...")
    model_model, model_sd3 = analyze_model_model_vs_sd3(score_tables["target_by_model"], scores)
    model_model.to_csv(os.path.join(output_dir, "04_model_model_agreement.csv"), index=False)
    model_sd3.to_csv(os.path.join(output_dir, "05_model_sd3_matched_agreement.csv"), index=False)

    print("Computing multitrait validity correlations...")
    validity = analyze_multitrait_validity(score_tables["all_long"], scores, criterion_reliability)
    validity.to_csv(os.path.join(output_dir, "06_multitrait_validity_correlations.csv"), index=False)

    validity_summary = analyze_matched_vs_mismatched(validity)
    validity_summary.to_csv(os.path.join(output_dir, "07_convergent_vs_discriminant_summary.csv"), index=False)

    print("Computing questionnaire benchmark correlations...")
    benchmarks = analyze_questionnaire_benchmarks(scores)
    benchmarks.to_csv(os.path.join(output_dir, "08_questionnaire_benchmark_correlations.csv"), index=False)

    print("Computing prompt specificity comparisons...")
    prompt_scores = pd.concat([
        score_tables["target_by_model_prompt"],
        score_tables["target_model_average_prompt"],
    ], ignore_index=True, sort=False)
    prompt_results = analyze_prompt_specificity(prompt_scores, scores, n_boot=n_boot, seed=seed)
    prompt_results.to_csv(os.path.join(output_dir, "09_prompt_specificity_broad_vs_specific.csv"), index=False)

    print("Computing social desirability analyses...")
    sd_discrepancy, sd_corr_compare = analyze_social_desirability(score_tables["all_long"], scores, n_boot=n_boot, seed=seed)
    sd_discrepancy.to_csv(os.path.join(output_dir, "10_mcsds_discrepancy_regressions.csv"), index=False)
    sd_corr_compare.to_csv(os.path.join(output_dir, "11_mcsds_correlation_comparisons.csv"), index=False)

    print("Running partial-rating sensitivity analysis...")
    # Primary analyses above use only rows with parse_success == True, meaning all three
    # trait ratings were recovered from an LLM call. This sensitivity analysis uses every
    # available trait-level rating, including partially parsed calls, and drops missing
    # trait cells row-wise after melting.
    melted_partial = melt_llm_ratings(llm, successful_only=False)
    score_tables_partial = make_participant_llm_scores(melted_partial)
    validity_partial = analyze_multitrait_validity(score_tables_partial["all_long"], scores, criterion_reliability)
    validity_partial.to_csv(os.path.join(output_dir, "12_sensitivity_partial_available_validity_correlations.csv"), index=False)
    validity_partial_summary = analyze_matched_vs_mismatched(validity_partial)
    validity_partial_summary.to_csv(os.path.join(output_dir, "13_sensitivity_partial_available_convergent_vs_discriminant.csv"), index=False)
    sd_disc_partial, sd_corr_partial = analyze_social_desirability(score_tables_partial["all_long"], scores, n_boot=n_boot, seed=seed)
    sd_disc_partial.to_csv(os.path.join(output_dir, "14_sensitivity_partial_available_mcsds_discrepancy.csv"), index=False)
    sd_corr_partial.to_csv(os.path.join(output_dir, "15_sensitivity_partial_available_mcsds_correlation_comparisons.csv"), index=False)

    # Save a compact README.
    readme = os.path.join(output_dir, "README_outputs.txt")
    with open(readme, "w", encoding="utf-8") as f:
        f.write("""D3 psychometric analysis outputs

00_data_quality_summary.csv
  Dataset size, participant counts, parse-success rates.

01_participant_llm_scores_long.csv
  Participant-level LLM scores, including individual models and MODEL_AVERAGE.

01_participant_model_average_scores_wide.csv
  Convenient merged participant-level file with model-average LLM scores and questionnaire scores.

02_within_model_repetition_icc.csv
  ICC(C,1)/ICC(C,k) and ICC(A,1)/ICC(A,k) across repeated calls within each model.

03_between_model_icc.csv
  ICC(C,1)/ICC(C,k) and ICC(A,1)/ICC(A,k) across models after averaging repetitions.

04_model_model_agreement.csv
  Pairwise model-model correlations for participant-level target-trait LLM scores.

05_model_sd3_matched_agreement.csv
  Each model's matched-trait correlation with SD3 scores.

06_multitrait_validity_correlations.csv
  Multitrait LLM-SD3 correlation matrix with raw and, if supplied, disattenuated correlations.

07_convergent_vs_discriminant_summary.csv
  Mean/median convergent vs discriminant correlations by model.

08_questionnaire_benchmark_correlations.csv
  Questionnaire-to-questionnaire correlation benchmarks from the same sample.

09_prompt_specificity_broad_vs_specific.csv
  Broad vs specific prompt validity comparisons using Williams dependent-correlation tests and bootstrap CIs.

10_mcsds_discrepancy_regressions.csv
  Regression of absolute standardized LLM-SD3 discrepancies on MCSDS.

11_mcsds_correlation_comparisons.csv
  Comparison of MCSDS-SD3 correlations with MCSDS-LLM correlations.

12_sensitivity_partial_available_validity_correlations.csv
  Same as output 06, but using all available trait-level ratings, including partially parsed calls.

13_sensitivity_partial_available_convergent_vs_discriminant.csv
  Same as output 07, using all available trait-level ratings.

14_sensitivity_partial_available_mcsds_discrepancy.csv
  Same as output 10, using all available trait-level ratings.

15_sensitivity_partial_available_mcsds_correlation_comparisons.csv
  Same as output 11, using all available trait-level ratings.

16_sd3_reliability_alphas.csv
  Optional output created when --item_long_csv is supplied; Cronbach's alpha for SD3 subscales.
""")

    if any(v is None for v in criterion_reliability.values()):
        print("\nNOTE: Some SD3 criterion reliability values are None.")
        print("Disattenuated correlations are reported as missing for those traits.")
        print("Edit CRITERION_RELIABILITY at the top of the script or pass --criterion_reliability_json.")

    print("\nAnalysis complete. Outputs written to:", output_dir)


def parse_reliability_json(s: Optional[str]) -> Dict[str, Optional[float]]:
    rel = dict(CRITERION_RELIABILITY)
    if s is None or str(s).strip() == "":
        return rel
    user = json.loads(s)
    for k, v in user.items():
        if k in rel:
            rel[k] = None if v is None else float(v)
    return rel


def main():
    parser = argparse.ArgumentParser(description="Analyze D3 LLM ratings and questionnaire scores.")
    parser.add_argument("--llm_csv", default=DEFAULT_LLM_CSV, help="Path to d3_llm_essay_ratings_long_FORMAL.csv")
    parser.add_argument("--scores_csv", default=DEFAULT_SCORES_CSV, help="Path to personality_scores_FORMAL.csv")
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR, help="Directory for output CSV files")
    parser.add_argument("--n_boot", type=int, default=N_BOOT, help="Number of bootstrap resamples")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    parser.add_argument(
        "--criterion_reliability_json",
        default=None,
        help='Optional JSON dict, e.g. \'{"Machiavellianism":0.75,"Narcissism":0.70,"Psychopathy":0.72}\''
    )
    parser.add_argument(
        "--item_long_csv",
        default=None,
        help="Optional stats_ready_long.csv-style file with SD3 item responses for computing sample Cronbach alphas."
    )
    args = parser.parse_args()

    rel = parse_reliability_json(args.criterion_reliability_json)
    run_analysis(args.llm_csv, args.scores_csv, args.output_dir, rel, n_boot=args.n_boot, seed=args.seed, item_long_csv=args.item_long_csv)


if __name__ == "__main__":
    main()
