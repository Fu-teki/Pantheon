
# %% STEP 0 - BUILD DATASET

import pandas as pd
from pathlib import Path
import re

CLEAN_DIR = Path(r"C:\Users\amint\Desktop\MASTER FTD\0_projects\NLP\scrapping_project\data\ecb_statements\ecb_statements_cleaned")

def load_statements(clean_dir: Path) -> pd.DataFrame:
    rows = []
    for fp in sorted(clean_dir.glob("*.txt")):
        # Extract YYYY-MM-DD from filename
        m = re.match(r"(\d{4}-\d{2}-\d{2})__", fp.name)
        if not m:
            continue
        dt = pd.to_datetime(m.group(1))
        text = fp.read_text(encoding="utf-8", errors="replace")
        rows.append({"date": dt, "file": fp.name, "text": text})
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df

df = load_statements(CLEAN_DIR)
print(df.shape)
df

# %% STEP 1 - COMMPUTE SIMILARITY (Jaccard on bigrams)

import re
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import PorterStemmer

stop = set(ENGLISH_STOP_WORDS)
stemmer = PorterStemmer()

def preprocess_tokens(text: str):
    tokens = re.findall(r"[A-Za-z]+", text.lower())
    tokens = [t for t in tokens if t not in stop]
    tokens = [stemmer.stem(t) for t in tokens]
    return tokens

def bigrams(tokens):
    return set(zip(tokens[:-1], tokens[1:]))

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

df["tokens"] = df["text"].apply(preprocess_tokens)
df["bigrams"] = df["tokens"].apply(bigrams)

sims = [None]
for i in range(1, len(df)):
    sims.append(jaccard(df.loc[i, "bigrams"], df.loc[i-1, "bigrams"]))
df["similarity"] = sims
df["log_similarity"] = df["similarity"].apply(lambda x: None if x is None or x <= 0 else float(pd.Series([x]).apply("log").iloc[0]))

# %% STEP 2 - COMPUTE PESSIMISM (Loughran–McDonald)

# %% STEP 3 - COMPUTE PESSIMISM (Loughran–McDonald)

from __future__ import annotations
from pathlib import Path
from typing import Tuple, Set, Optional
import re
import pandas as pd


def add_pessimism_from_lm_master_dictionary(
    df: pd.DataFrame,
    lm_csv_path: str | Path,
    text_col: str = "text",
    output_col: str = "pessimism",
    output_pos_col: str = "pos_count",
    output_neg_col: str = "neg_count",
    output_total_col: str = "total_words",
    remove_qa: bool = True,          # <—— USER OPTION
    qa_marker: str = "Question:",    # <—— Q&A START MARKER
) -> Tuple[pd.DataFrame, Set[str], Set[str]]:
    """
    Computes dictionary-based pessimism:

        Pessimism_i = (NegativeWords_i - PositiveWords_i) / TotalWords_i

    Methodologically faithful to Amaya & Filbien (2015):
    - Positive/Negative words counted on RAW lowercase tokens (no stemming)
    - TotalWords_i computed on RAW text
    - Q&A transcript optionally removed (recommended)

    Parameters
    ----------
    df : pd.DataFrame
        Must contain df[text_col] with raw statement text.
    lm_csv_path : str | Path
        Path to Loughran–McDonald Master Dictionary CSV.
    remove_qa : bool
        If True, removes everything starting at first occurrence of qa_marker.
    qa_marker : str
        Marker indicating start of Q&A (default: "Question:").

    Returns
    -------
    (df_out, positive_set, negative_set)
    """

    lm_csv_path = Path(lm_csv_path)
    if not lm_csv_path.exists():
        raise FileNotFoundError(f"LM dictionary CSV not found: {lm_csv_path}")

    lm = pd.read_csv(lm_csv_path)

    # Build LM word sets (year-coded indicator: value > 0)
    positive_set = set(
        lm.loc[lm["Positive"] > 0, "Word"].astype(str).str.lower()
    )
    negative_set = set(
        lm.loc[lm["Negative"] > 0, "Word"].astype(str).str.lower()
    )

    if not positive_set or not negative_set:
        raise ValueError("Empty LM sentiment sets — check dictionary file.")

    word_re = re.compile(r"[A-Za-z]+")

    def preprocess_text(text: Optional[str]) -> str:
        if not isinstance(text, str):
            return ""

        t = text

        # Remove Q&A if requested
        if remove_qa:
            idx = t.find(qa_marker)
            if idx != -1:
                t = t[:idx]

        return t

    def score(text: Optional[str]) -> tuple[int, int, int, float]:
        t = preprocess_text(text)
        tokens = word_re.findall(t.lower())
        total = len(tokens)

        if total == 0:
            return 0, 0, 0, 0.0

        pos = sum(1 for w in tokens if w in positive_set)
        neg = sum(1 for w in tokens if w in negative_set)

        pess = (neg - pos) / total
        return pos, neg, total, pess

    df_out = df.copy()
    scores = df_out[text_col].apply(score).apply(pd.Series)
    scores.columns = [output_pos_col, output_neg_col, output_total_col, output_col]

    df_out[[output_pos_col, output_neg_col, output_total_col, output_col]] = scores

    return df_out, positive_set, negative_set

# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    df, pos_set, neg_set = add_pessimism_from_lm_master_dictionary(
        df,
        lm_csv_path="data/Loughran-McDonald_MasterDictionary_1993-2024.csv",
        remove_qa=True,
        qa_marker="Question:",
    )

    print(df["pessimism"].describe())

# %% STEP 4 - EVENT STUDY WITH DROPOUT MONITORING

from __future__ import annotations

from datetime import timedelta
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import yfinance as yf


def compute_abs_car_constant_mean_with_monitoring(
    df: pd.DataFrame,
    date_col: str = "date",
    # index_ticker: str = "^STOXX50E",
    index_ticker: str = "FEZ",
    estimation_window: Tuple[int, int] = (-120, -20),
    event_window: Tuple[int, int] = (-1, 1),
    download_buffer_days: int = 250,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """
    Computes CAR and |CAR| (constant-mean model) AND logs why observations are dropped.

    Dropout reasons monitored (mutually exclusive, in this priority order):
      1) OUTSIDE_YF_COVERAGE: ECB date is before first return or after last return.
      2) INSUFFICIENT_PRE_EVENT_TRADING_DAYS: not enough trading days to fill estimation window.
      3) INCOMPLETE_EVENT_WINDOW: not enough trading days to fill event window.
      4) MISSING_MARKET_DATA: event date not alignable / unexpected issue.

    Returns
    -------
    df_out : pd.DataFrame
        Original df + columns: car, abs_car, event_date_used, dropout_reason
    monitor_df : pd.DataFrame
        One row per event with detailed diagnostics.
    counts : dict
        Counts by dropout_reason (including "OK").
    """
    df_out = df.copy()
    df_out[date_col] = pd.to_datetime(df_out[date_col])

    # Download a sufficiently wide range around all ECB dates
    start = df_out[date_col].min() - timedelta(days=download_buffer_days)
    end = df_out[date_col].max() + timedelta(days=download_buffer_days)

    px = yf.download(index_ticker, start=start, end=end, progress=False, auto_adjust=False)["Adj Close"].dropna().squeeze()
    rets = px.pct_change().dropna()

    if rets.empty:
        raise RuntimeError("No returns downloaded from Yahoo Finance. Check ticker or connectivity.")

    first_ret_date = rets.index.min()
    last_ret_date = rets.index.max()

    print("Yahoo Finance Euro Stoxx 50 coverage:")
    print("  First available trading day:", first_ret_date.date())
    print("  Last available trading day :", last_ret_date.date())
    print("  Total trading days available:", len(rets))

    # Prepare outputs
    car_vals: List[float] = []
    abs_car_vals: List[float] = []
    used_dates: List[pd.Timestamp] = []
    reasons: List[str] = []
    monitor_rows: List[dict] = []

    counts: Dict[str, int] = {
        "OK": 0,
        "OUTSIDE_YF_COVERAGE": 0,
        "INSUFFICIENT_PRE_EVENT_TRADING_DAYS": 0,
        "INCOMPLETE_EVENT_WINDOW": 0,
        "MISSING_MARKET_DATA": 0,
    }

    for event_dt in df_out[date_col]:
        row = {
            "event_date_original": event_dt,
            "event_date_used": pd.NaT,
            "reason": None,
            "first_ret_date": first_ret_date,
            "last_ret_date": last_ret_date,
            "event_loc": np.nan,
            "est_start_loc": np.nan,
            "est_end_loc": np.nan,
            "ev_start_loc": np.nan,
            "ev_end_loc": np.nan,
        }

        # 4) ECB date outside YF coverage
        # (If event date is after last return date, searchsorted will point beyond last index.)
        if event_dt < first_ret_date or event_dt > last_ret_date:
            reason = "OUTSIDE_YF_COVERAGE"
            counts[reason] += 1
            car_vals.append(np.nan)
            abs_car_vals.append(np.nan)
            used_dates.append(pd.NaT)
            reasons.append(reason)
            row["reason"] = reason
            monitor_rows.append(row)
            continue

        # Align event date to trading day (next trading day if needed)
        try:
            loc = rets.index.get_indexer([event_dt], method="bfill")[0]
        except Exception:
            loc = -1

        if loc is None or loc < 0 or loc >= len(rets):
            reason = "MISSING_MARKET_DATA"
            counts[reason] += 1
            car_vals.append(np.nan)
            abs_car_vals.append(np.nan)
            used_dates.append(pd.NaT)
            reasons.append(reason)
            row["reason"] = reason
            monitor_rows.append(row)
            continue

        event_used = rets.index[loc]

        # Window indices
        est_start = loc + estimation_window[0]
        est_end = loc + estimation_window[1]
        ev_start = loc + event_window[0]
        ev_end = loc + event_window[1]

        row.update({
            "event_date_used": event_used,
            "event_loc": int(loc),
            "est_start_loc": int(est_start),
            "est_end_loc": int(est_end),
            "ev_start_loc": int(ev_start),
            "ev_end_loc": int(ev_end),
        })

        # 1) Not enough pre-event trading days for estimation window
        if est_start < 0:
            reason = "INSUFFICIENT_PRE_EVENT_TRADING_DAYS"
            counts[reason] += 1
            car_vals.append(np.nan)
            abs_car_vals.append(np.nan)
            used_dates.append(event_used)
            reasons.append(reason)
            row["reason"] = reason
            monitor_rows.append(row)
            continue

        # 2) Event window incomplete (needs data up to ev_end)
        if ev_end >= len(rets):
            reason = "INCOMPLETE_EVENT_WINDOW"
            counts[reason] += 1
            car_vals.append(np.nan)
            abs_car_vals.append(np.nan)
            used_dates.append(event_used)
            reasons.append(reason)
            row["reason"] = reason
            monitor_rows.append(row)
            continue

        # Compute CAR (constant mean)
        mu = rets.iloc[est_start:est_end + 1].mean()
        ar = rets.iloc[ev_start:ev_end + 1] - mu
        car = float(ar.sum())

        car_vals.append(car)
        abs_car_vals.append(abs(car))
        used_dates.append(event_used)
        reasons.append("OK")
        counts["OK"] += 1

        row["reason"] = "OK"
        monitor_rows.append(row)

    df_out["event_date_used"] = used_dates
    df_out["car"] = car_vals
    df_out["abs_car"] = abs_car_vals
    df_out["dropout_reason"] = reasons

    monitor_df = pd.DataFrame(monitor_rows)

    return df_out, monitor_df, counts


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    df, monitor_df, counts = compute_abs_car_constant_mean_with_monitoring(df)

    print("Counts by reason:")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    # See a quick breakdown
    print("\nReason share (%):")
    total = sum(counts.values())
    for k, v in counts.items():
        print(f"  {k}: {100*v/total:.1f}%")

    # Inspect first few dropouts
    print("\nExamples of dropouts:")
    print(monitor_df.loc[monitor_df["reason"] != "OK", ["event_date_original", "event_date_used", "reason"]])

    # If you want: save monitoring report
    # monitor_df.to_csv("data/event_study_dropout_monitor.csv", index=False)
    

# %% STEP 5 - MAIN REGRESSION: |CAR| ON PESSIMISM AND SIMILARITY

import pandas as pd
import statsmodels.api as sm


def run_main_regression(df: pd.DataFrame):
    """
    Runs the main regression:

        |CAR| = alpha + beta1 * Pessimism + beta2 * log(Similarity) + error

    Uses heteroskedasticity-robust (HC1) standard errors.
    """

    required_cols = ["abs_car", "pessimism", "log_similarity"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")

    # Drop missing observations
    reg_df = df[required_cols].dropna().copy()

    # Dependent variable
    y = reg_df["abs_car"]

    # Independent variables
    X = reg_df[["pessimism", "log_similarity"]]
    X = sm.add_constant(X)

    # OLS with robust SE
    model = sm.OLS(y, X).fit(cov_type="HC1")

    return model

if __name__ == "__main__":
    model = run_main_regression(df)
    print(model.summary())

# %% STEP 6 - PAPER VISUALS: FIG 1 + TABLES 2–4 (Amaya & Filbien 2015-style)

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# -----------------------------
# Helpers: formatting + stars
# -----------------------------
def _stars(p: float) -> str:
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


def _fmt_coef(coef: float, p: float, decimals: int = 3) -> str:
    if coef is None or (isinstance(coef, float) and (math.isnan(coef) or math.isinf(coef))):
        return "NA"
    return f"{coef:.{decimals}f}{_stars(p)}"


def _fmt_r2(adj_r2: float) -> str:
    if adj_r2 is None or (isinstance(adj_r2, float) and math.isnan(adj_r2)):
        return "NA"
    # paper reports % for adjusted R2
    return f"{adj_r2 * 100:.2f}%"


def _fit_ols_hc1(df: pd.DataFrame, y: str, x: Sequence[str]) -> Optional[sm.regression.linear_model.RegressionResultsWrapper]:
    cols = [y, *x]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        return None

    reg = df[cols].dropna().copy()
    if len(reg) < 20:
        return None

    Y = reg[y].astype(float)
    X = reg[list(x)].astype(float)
    X = sm.add_constant(X)

    return sm.OLS(Y, X).fit(cov_type="HC1")


# -----------------------------
# Fig 1: Similarity over time
# -----------------------------
def plot_figure_1_similarity(
    df: pd.DataFrame,
    date_col: str = "date",
    similarity_col: str = "similarity",
    title: str = "Fig. 1. Similarity measure since ECB’s creation.",
) -> None:
    """
    Recreates Fig. 1 style: line plot of Similarity over time.
    """
    for c in [date_col, similarity_col]:
        if c not in df.columns:
            raise KeyError(f"Missing required column: {c}")

    tmp = df[[date_col, similarity_col]].dropna().copy()
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp = tmp.sort_values(date_col)

    plt.figure(figsize=(10, 6))
    plt.plot(tmp[date_col], tmp[similarity_col], linewidth=1.0, label="Similarity")
    plt.ylabel("Similarity")
    plt.xlabel("")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()


# -----------------------------
# Table 2: Summary statistics
# -----------------------------
def make_table_2_summary_statistics(
    df: pd.DataFrame,
    mapping: Optional[Dict[str, str]] = None,
    scale_percent_vars: Optional[Sequence[str]] = ("CAR", "|CAR|", "Pessimism"),
) -> pd.DataFrame:
    """
    Creates Table 2-style summary statistics:
      Mean, Std. dev., Min., Quartile 1, Median, Quartile 3, Max.

    mapping: {paper_label -> df_column}
      Default uses your current df: CAR=car, |CAR|=abs_car, Pessimism=pessimism, Similarity=similarity
      And optionally: Output gap, Inflation, Delta MRO.

    scale_percent_vars: rows to multiply by 100 (paper reports in percent).
    """
    default_mapping = {
        "CAR": "car",
        "|CAR|": "abs_car",
        "Pessimism": "pessimism",
        "Similarity": "similarity",
        "Output gap": "output_gap",     # optional
        "Inflation": "inflation",       # optional
        "Delta MRO": "delta_mro",        # optional
    }
    if mapping:
        default_mapping.update(mapping)

    rows = []
    for label, col in default_mapping.items():
        if col not in df.columns:
            continue
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue

        # paper reports CAR/|CAR|/Pessimism in percent units
        if scale_percent_vars and label in set(scale_percent_vars):
            s = s * 100.0

        rows.append(
            {
                "Variables": label,
                "Mean": s.mean(),
                "Std. dev.": s.std(ddof=1),
                "Min.": s.min(),
                "Quartile 1": s.quantile(0.25),
                "Median": s.quantile(0.50),
                "Quartile 3": s.quantile(0.75),
                "Max.": s.max(),
            }
        )

    out = pd.DataFrame(rows)
    # paper order
    paper_order = ["CAR", "|CAR|", "Pessimism", "Similarity", "Output gap", "Inflation", "Delta MRO"]
    out["__order"] = out["Variables"].apply(lambda x: paper_order.index(x) if x in paper_order else 999)
    out = out.sort_values("__order").drop(columns="__order").reset_index(drop=True)

    # round like paper (2 decimals)
    for c in out.columns:
        if c != "Variables":
            out[c] = out[c].astype(float).round(2)
    return out


# -----------------------------
# Table 3: Similarity regressions
# -----------------------------
def make_table_3_similarity_regressions(
    df: pd.DataFrame,
    date_col: str = "date",
    similarity_col: str = "similarity",
    output_gap_col: str = "output_gap",
    inflation_col: str = "inflation",
    delta_mro_col: str = "delta_mro",
) -> pd.DataFrame:
    """
    Paper Table 3 structure (columns 1–4):

      (1) Similarity ~ Output gap + Inflation + Delta MRO
      (2) Similarity ~ Time (continuous trend)
      (3) Similarity ~ Time + Output gap + Inflation + Delta MRO
      (4) Similarity ~ Time(count) + Output gap + Inflation + Delta MRO

    Notes:
    - If macro/MRO columns are not present, models requiring them will show NA.
    - Uses HC1 robust SE.
    """
    tmp = df.copy()
    if date_col not in tmp.columns:
        raise KeyError(f"Missing {date_col}")
    tmp[date_col] = pd.to_datetime(tmp[date_col])
    tmp = tmp.sort_values(date_col).reset_index(drop=True)

    # Time (continuous): years since first statement
    t0 = tmp[date_col].min()
    tmp["Time"] = (tmp[date_col] - t0).dt.days / 365.25

    # Time(count): 1..N
    tmp["Time(count)"] = np.arange(1, len(tmp) + 1, dtype=float)

    # Fit models
    models = {
        "(1)": _fit_ols_hc1(tmp, similarity_col, [output_gap_col, inflation_col, delta_mro_col]),
        "(2)": _fit_ols_hc1(tmp, similarity_col, ["Time"]),
        "(3)": _fit_ols_hc1(tmp, similarity_col, ["Time", output_gap_col, inflation_col, delta_mro_col]),
        "(4)": _fit_ols_hc1(tmp, similarity_col, ["Time(count)", output_gap_col, inflation_col, delta_mro_col]),
    }

    # Paper rows
    row_vars = ["Intercept", "Time", "Time(count)", "Output gap", "Inflation", "Delta MRO", "Adjusted R$^2$"]
    out = pd.DataFrame({"Variable": row_vars})

    def grab(m, var):
        if m is None:
            return "NA"
        if var == "Adjusted R$^2$":
            return _fmt_r2(m.rsquared_adj)
        if var == "Intercept":
            coef = float(m.params.get("const", np.nan))
            p = float(m.pvalues.get("const", np.nan))
            return _fmt_coef(coef, p)
        # map displayed names to regressor names
        key = var
        coef = float(m.params.get(key, np.nan))
        p = float(m.pvalues.get(key, np.nan))
        return _fmt_coef(coef, p)

    for col, m in models.items():
        out[col] = [grab(m, v) for v in row_vars]

    return out


# -----------------------------
# Table 4: |CAR| regressions
# -----------------------------
def make_table_4_abs_car_regressions(
    df: pd.DataFrame,
    abs_car_col: str = "abs_car",
    pessimism_col: str = "pessimism",
    similarity_col: str = "similarity",
    output_gap_col: str = "output_gap",
    inflation_col: str = "inflation",
    delta_mro_col: str = "delta_mro",
    surprise_mro_col: str = "surprise_mro",
    scale_percent_y: bool = True,
) -> pd.DataFrame:
    """
    Paper Table 4 structure (columns 1–5), using |CAR| as dependent variable:

      (1) |CAR| ~ Pessimism
      (2) |CAR| ~ Output gap + Inflation + Delta MRO
      (3) |CAR| ~ Pessimism + (Pessimism × similarity)
      (4) |CAR| ~ Pessimism + (Pessimism × similarity) + Output gap + Inflation + Delta MRO
      (5) |CAR| ~ Pessimism + (Pessimism × similarity) + Output gap + Inflation + Delta MRO + Surprise MRO

    Notes:
    - If some columns are missing, that model returns NA.
    - Uses HC1 robust SE.
    - Paper reports |CAR| in percent; set scale_percent_y=True to multiply y by 100 for table only.
    """
    tmp = df.copy()

    if abs_car_col not in tmp.columns:
        raise KeyError(f"Missing {abs_car_col}")

    # For table presentation: scale dependent variable to percent units (like paper)
    if scale_percent_y:
        tmp["__abs_car_pct__"] = pd.to_numeric(tmp[abs_car_col], errors="coerce") * 100.0
        y_col = "__abs_car_pct__"
    else:
        y_col = abs_car_col

    # interaction
    if pessimism_col in tmp.columns and similarity_col in tmp.columns:
        tmp["Pessimism × similarity"] = pd.to_numeric(tmp[pessimism_col], errors="coerce") * pd.to_numeric(tmp[similarity_col], errors="coerce")

    models = {
        "(1)": _fit_ols_hc1(tmp, y_col, [pessimism_col]),
        "(2)": _fit_ols_hc1(tmp, y_col, [output_gap_col, inflation_col, delta_mro_col]),
        "(3)": _fit_ols_hc1(tmp, y_col, [pessimism_col, "Pessimism × similarity"]),
        "(4)": _fit_ols_hc1(tmp, y_col, [pessimism_col, "Pessimism × similarity", output_gap_col, inflation_col, delta_mro_col]),
        "(5)": _fit_ols_hc1(tmp, y_col, [pessimism_col, "Pessimism × similarity", output_gap_col, inflation_col, delta_mro_col, surprise_mro_col]),
    }

    row_vars = [
        "Intercept",
        "Pessimism",
        "Pessimism × similarity",
        "Output gap",
        "Inflation",
        "Delta MRO",
        "Surprise MRO",
        "Adjusted R$^2$",
    ]
    out = pd.DataFrame({"Variable": row_vars})

    def grab(m, var):
        if m is None:
            return "NA"
        if var == "Adjusted R$^2$":
            return _fmt_r2(m.rsquared_adj)
        if var == "Intercept":
            coef = float(m.params.get("const", np.nan))
            p = float(m.pvalues.get("const", np.nan))
            return _fmt_coef(coef, p)
        key = var
        # map display names to df columns used in regressions
        if var == "Pessimism":
            key = pessimism_col
        if var == "Output gap":
            key = output_gap_col
        if var == "Inflation":
            key = inflation_col
        if var == "Delta MRO":
            key = delta_mro_col
        if var == "Surprise MRO":
            key = surprise_mro_col

        coef = float(m.params.get(key, np.nan))
        p = float(m.pvalues.get(key, np.nan))
        return _fmt_coef(coef, p)

    for col, m in models.items():
        out[col] = [grab(m, v) for v in row_vars]

    return out


# -----------------------------
# One wrapper to "display" all
# -----------------------------
def display_paper_visuals(
    df: pd.DataFrame,
    date_col: str = "date",
    similarity_col: str = "similarity",
    pessimism_col: str = "pessimism",
    car_col: str = "car",
    abs_car_col: str = "abs_car",
    output_gap_col: str = "output_gap",
    inflation_col: str = "inflation",
    delta_mro_col: str = "delta_mro",
    surprise_mro_col: str = "surprise_mro",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Produces:
      - Fig 1 plot
      - Table 2 DataFrame
      - Table 3 DataFrame
      - Table 4 DataFrame

    Returns (table2, table3, table4).
    """
    # Fig 1
    plot_figure_1_similarity(df, date_col=date_col, similarity_col=similarity_col)

    # Table 2
    table2 = make_table_2_summary_statistics(
        df,
        mapping={
            "CAR": car_col,
            "|CAR|": abs_car_col,
            "Pessimism": pessimism_col,
            "Similarity": similarity_col,
            "Output gap": output_gap_col,
            "Inflation": inflation_col,
            "Delta MRO": delta_mro_col,
        },
    )

    # Table 3
    table3 = make_table_3_similarity_regressions(
        df,
        date_col=date_col,
        similarity_col=similarity_col,
        output_gap_col=output_gap_col,
        inflation_col=inflation_col,
        delta_mro_col=delta_mro_col,
    )

    # Table 4
    table4 = make_table_4_abs_car_regressions(
        df,
        abs_car_col=abs_car_col,
        pessimism_col=pessimism_col,
        similarity_col=similarity_col,
        output_gap_col=output_gap_col,
        inflation_col=inflation_col,
        delta_mro_col=delta_mro_col,
        surprise_mro_col=surprise_mro_col,
        scale_percent_y=True,  # paper units
    )

    return table2, table3, table4


# -------------------------
# Example usage
# -------------------------
if __name__ == "__main__":
    table2, table3, table4 = display_paper_visuals(df)

    print("\nTABLE 2 (summary stats):")
    print(table2)

    print("\nTABLE 3 (similarity regressions):")
    print(table3)

    print("\nTABLE 4 (|CAR| regressions):")
    print(table4)

# %% STEP 6 - VISUAL REPLICATION HELPERS — Fig. 1 and Tables 2–4 (Amaya & Filbien, 2015)

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


# -----------------------------
# Utility helpers
# -----------------------------

def _ensure_datetime(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out = out.sort_values(date_col)
    return out


def _add_time_trend(df: pd.DataFrame, date_col: str, trend_col: str = "time_trend") -> pd.DataFrame:
    out = _ensure_datetime(df, date_col)
    # "Time" in the paper is effectively a linear trend (1..T) across announcements
    out[trend_col] = np.arange(1, len(out) + 1, dtype=float)
    return out


def _ols_hc1(y: pd.Series, X: pd.DataFrame):
    Xc = sm.add_constant(X, has_constant="add")
    model = sm.OLS(y, Xc).fit(cov_type="HC1")
    return model


def _format_reg_table(models: Dict[str, sm.regression.linear_model.RegressionResultsWrapper],
                      var_order: List[str],
                      decimals: int = 3) -> pd.DataFrame:
    """
    Create a compact regression table similar to paper tables:
    cells = coef + stars, and (robust SE) on next line.
    """
    def stars(p):
        if p < 0.01: return "***"
        if p < 0.05: return "**"
        if p < 0.10: return "*"
        return ""

    cols = list(models.keys())
    rows = []
    for v in ["const"] + var_order:
        row_coef = []
        row_se = []
        for name in cols:
            m = models[name]
            if v in m.params.index:
                coef = m.params[v]
                se = m.bse[v]
                p = m.pvalues[v]
                row_coef.append(f"{coef:.{decimals}f}{stars(p)}")
                row_se.append(f"({se:.{decimals}f})")
            else:
                row_coef.append("")
                row_se.append("")
        rows.append((("Intercept" if v == "const" else v), row_coef))
        rows.append(("", row_se))

    # Add summary lines
    r2 = ["{:.3f}".format(models[c].rsquared_adj) for c in cols]
    nobs = [str(int(models[c].nobs)) for c in cols]
    rows.append(("Adj. R²", r2))
    rows.append(("N", nobs))

    # Build DataFrame
    out = pd.DataFrame(
        {col: [r[1][i] for r in rows] for i, col in enumerate(cols)},
        index=[r[0] for r in rows]
    )
    return out


# ============================================================
# FIGURE 1 — Similarity across time
# ============================================================

import pandas as pd
import plotly.graph_objects as go


def plot_fig1_similarity_over_time(
    df: pd.DataFrame,
    date_col: str = "date",
    similarity_col: str = "similarity",
    rolling_window: int = 12,
    title: str = "Fig. 1 — Evolution of ECB communication similarity over time",
) -> go.Figure:
    """
    Plotly version of Fig. 1: similarity of ECB communication over time.
    Displays raw similarity and a rolling mean.
    """

    # Ensure datetime
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")

    if similarity_col not in d.columns:
        raise KeyError(f"'{similarity_col}' not found in df. Available: {list(d.columns)}")

    s = pd.to_numeric(d[similarity_col], errors="coerce")
    if s.isna().all():
        raise ValueError(f"'{similarity_col}' could not be converted to numeric.")

    roll = s.rolling(
        rolling_window,
        min_periods=max(3, rolling_window // 3)
    ).mean()

    fig = go.Figure()

    # Raw similarity
    fig.add_trace(
        go.Scatter(
            x=d[date_col],
            y=s,
            mode="lines",
            name=similarity_col,
            line=dict(width=1),
        )
    )

    # Rolling mean
    fig.add_trace(
        go.Scatter(
            x=d[date_col],
            y=roll,
            mode="lines",
            name=f"{rolling_window}-announcement rolling mean",
            line=dict(width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Similarity (cosine)",
        template="plotly_dark",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=60, r=20, t=60, b=50),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig

# ============================================================
# TABLE 2 — Descriptive statistics
# ============================================================

def make_table2_descriptive_stats(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    rename: Optional[Dict[str, str]] = None,
    percent_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Creates a Table-2-style descriptive stats table: mean, std, min, p25, median, p75, max, N.

    In the paper, Table 2 includes variables like CAR, |CAR|, Pessimism, Inflation, Output gap, etc.
    Your df currently includes: car, abs_car, pessimism, similarity, log_similarity, etc.

    Parameters
    ----------
    columns : list[str], optional
        Which df columns to include. If None, uses a sensible default based on availability.
    rename : dict, optional
        Column display names.
    percent_cols : list[str], optional
        Columns to display as percents (multiplied by 100). Example: ["car","abs_car","pessimism"]

    Returns
    -------
    pd.DataFrame
        A formatted descriptive statistics table.
    """
    default = ["car", "abs_car", "pessimism", "similarity", "log_similarity", "total_words"]
    if columns is None:
        columns = [c for c in default if c in df.columns]

    if not columns:
        raise ValueError("No columns selected for descriptive statistics.")

    d = df[columns].copy()
    for c in columns:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    if percent_cols:
        for c in percent_cols:
            if c in d.columns:
                d[c] = d[c] * 100.0

    stats = d.describe(percentiles=[0.25, 0.50, 0.75]).T
    stats = stats.rename(columns={
        "count": "N",
        "mean": "Mean",
        "std": "Std",
        "min": "Min",
        "25%": "P25",
        "50%": "Median",
        "75%": "P75",
        "max": "Max",
    })

    # Keep ordering
    stats = stats[["N", "Mean", "Std", "Min", "P25", "Median", "P75", "Max"]]

    if rename:
        stats.index = [rename.get(i, i) for i in stats.index]

    return stats


# ============================================================
# TABLE 3 — Explaining similarity with time (OLS)
# ============================================================

def make_table3_similarity_regressions(
    df: pd.DataFrame,
    date_col: str = "date",
    dep_col: str = "similarity",
    controls_sets: Optional[List[List[str]]] = None,
) -> pd.DataFrame:
    """
    Replicates the logic of Table 3:
      Similarity_t = a + b*Time + controls + e

    The paper includes macro controls (Output gap, Inflation) and policy move (DeltaMRO).
    Your df may not include these; this function will:
      - include whatever you pass in controls_sets AND exists in df,
      - always include time trend.

    Returns a formatted regression table with robust (HC1) SE.
    """
    d = _add_time_trend(df, date_col, trend_col="time_trend")

    if dep_col not in d.columns:
        raise KeyError(f"'{dep_col}' not found in df.")

    y = pd.to_numeric(d[dep_col], errors="coerce")

    if controls_sets is None:
        # Paper-like progression (only included if present)
        controls_sets = [
            [],  # (1) Time only
            ["outputgap", "inflation", "delta_mro"],  # (2) macro/policy controls (names may differ in your df)
        ]

    models = {}
    for i, controls in enumerate(controls_sets, start=1):
        # Always include time trend
        Xcols = ["time_trend"] + [c for c in controls if c in d.columns]
        reg = pd.concat([y, d[Xcols]], axis=1).dropna()
        y_i = reg[dep_col]
        X_i = reg[Xcols]
        models[f"({i})"] = _ols_hc1(y_i, X_i)

    # Variable order in table output (excluding const)
    var_order = ["time_trend"] + sorted({c for cs in controls_sets for c in cs if c in d.columns})
    out = _format_reg_table(models, var_order=var_order, decimals=3)

    # Improve labels
    out = out.rename(index={
        "time_trend": "Time (trend)",
        "outputgap": "Output gap",
        "inflation": "Inflation",
        "delta_mro": "ΔMRO",
    })
    return out


# ============================================================
# TABLE 4 — Explaining |CAR| with pessimism and similarity (OLS)
# ============================================================

def make_table4_abs_car_regressions(
    df: pd.DataFrame,
    dep_col: str = "abs_car",
    pessimism_col: str = "pessimism",
    similarity_col_for_interaction: str = "similarity",
    similarity_col_for_level: str = "log_similarity",
    model_specs: Optional[List[Dict[str, List[str]]]] = None,
) -> pd.DataFrame:
    """
    Replicates the logic of Table 4:
      |CAR| = a + b*Pessimism + c*(Pessimism x Similarity) + controls + e

    The paper shows several columns (specifications). Your df lacks macro/policy controls,
    so by default this function builds the paper's core columns that you CAN estimate:
      (1) |CAR| ~ Pessimism
      (2) |CAR| ~ Pessimism + log_similarity
      (3) |CAR| ~ Pessimism + (Pessimism x similarity) + log_similarity

    You can override via model_specs.

    Returns a formatted regression table with robust (HC1) SE.
    """
    if dep_col not in df.columns:
        raise KeyError(f"'{dep_col}' not found in df.")
    if pessimism_col not in df.columns:
        raise KeyError(f"'{pessimism_col}' not found in df.")
    if similarity_col_for_level not in df.columns and similarity_col_for_interaction not in df.columns:
        raise KeyError("Need at least one similarity column for Table 4.")

    d = df.copy()
    d[dep_col] = pd.to_numeric(d[dep_col], errors="coerce")
    d[pessimism_col] = pd.to_numeric(d[pessimism_col], errors="coerce")

    # Interaction term (paper: Pessimism × Similarity level)
    if similarity_col_for_interaction in d.columns:
        d["pess_x_sim"] = pd.to_numeric(d[similarity_col_for_interaction], errors="coerce") * d[pessimism_col]
    else:
        d["pess_x_sim"] = np.nan

    if model_specs is None:
        model_specs = [
            {"X": [pessimism_col]},  # (1)
            {"X": [pessimism_col, similarity_col_for_level] if similarity_col_for_level in d.columns else [pessimism_col]},  # (2)
            {"X": [pessimism_col, "pess_x_sim"] + ([similarity_col_for_level] if similarity_col_for_level in d.columns else [])},  # (3)
        ]

    models = {}
    all_vars = []
    for i, spec in enumerate(model_specs, start=1):
        Xcols = [c for c in spec["X"] if c in d.columns]
        reg = d[[dep_col] + Xcols].dropna()
        y = reg[dep_col]
        X = reg[Xcols]
        models[f"({i})"] = _ols_hc1(y, X)
        all_vars.extend(Xcols)

    # Choose variable order (excluding const)
    var_order = []
    for v in [pessimism_col, "pess_x_sim", similarity_col_for_level]:
        if v in set(all_vars):
            var_order.append(v)
    # plus any other controls passed
    for v in all_vars:
        if v not in var_order:
            var_order.append(v)

    out = _format_reg_table(models, var_order=var_order, decimals=3)

    # Improve labels
    out = out.rename(index={
        pessimism_col: "Pessimism",
        "pess_x_sim": "Pessimism × Similarity",
        similarity_col_for_level: "Log similarity",
    })
    return out


# ============================================================
# Example usage (run these in your notebook)
# ============================================================
if __name__ == "__main__":
    # Fig. 1
    fig = plot_fig1_similarity_over_time(df, date_col="date", similarity_col="similarity", rolling_window=12)
    fig.show()

    # Table 2 (paper-like: show % for returns and pessimism)
    t2 = make_table2_descriptive_stats(
        df,
        columns=["car", "abs_car", "pessimism", "similarity", "log_similarity", "total_words"],
        rename={
            "car": "CAR",
            "abs_car": "|CAR|",
            "pessimism": "Pessimism",
            "similarity": "Similarity",
            "log_similarity": "Log similarity",
            "total_words": "Total words",
        },
        percent_cols=["car", "abs_car", "pessimism"],
    )
    print("\nTABLE 2 — Descriptive statistics (returns/pessimism shown in %):\n")
    print(t2)

    # Table 3 (Similarity ~ Time trend [+ optional controls if you add columns later])
    t3 = make_table3_similarity_regressions(
        df,
        date_col="date",
        dep_col="similarity",
        # If later you add macro/policy cols, list them here using your actual column names:
        controls_sets=[[], ["pessimism"]],  # placeholder example; replace with macro controls if available
    )
    print("\nTABLE 3 — Explaining similarity with time (OLS, HC1):\n")
    print(t3)

    # Table 4 (|CAR| ~ Pessimism, + log_similarity, + interaction)
    t4 = make_table4_abs_car_regressions(
        df,
        dep_col="abs_car",
        pessimism_col="pessimism",
        similarity_col_for_interaction="similarity",
        similarity_col_for_level="log_similarity",
    )
    print("\nTABLE 4 — Explaining |CAR| (OLS, HC1):\n")
    print(t4)



# %%
