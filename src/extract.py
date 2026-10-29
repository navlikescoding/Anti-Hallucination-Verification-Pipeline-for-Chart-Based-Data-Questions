"""
extract.py — reading a chart IMAGE back into a data table (Section 3b).

When a user only has a chart picture, the oracle has no table to compute from.
So a vision model first reads the chart into a table. That reading can itself be
wrong, so we:
  1. read every chart with TWO different vision models,
  2. cross-check the two tables value by value ("confirmed" vs "uncertain"),
  3. score each reading against ChartQA's true table, to measure how often the
     readings are wrong and whether the cross-check catches it.
"""
import base64
import csv
import io
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from . import config
from .config import TEST_DIR
from .data_utils import clean_label, table_from_raw, to_number
from .oracle import mentioned, oracle, pick_series
from .llm import ask, STYLE_INSTRUCTIONS

# ===================================================================== 1. prompts
EXTRACT_SYSTEM = ("You convert chart images into data tables. Report exactly what the chart "
                  "shows. Never invent categories or values that are not in the chart.")

EXTRACT_PROMPT = (
    "Extract the data shown in this chart as a CSV table.\n"
    "- First column: the category or x-axis labels, exactly as written in the chart.\n"
    "- One further column per data series, with the series name (or legend entry) as its header.\n"
    "- Numbers only in the value cells: no units, no % signs, no thousands separators.\n"
    "- If a value is printed on the chart, copy it exactly; otherwise estimate it from the axis.\n"
    "Output ONLY the CSV, inside a ```csv code block."
)

ANSWER_SYSTEM = ("You answer questions about chart images for blind and low-vision users. "
                 "Answer using only what the chart shows.")


def answer_prompt(question: str) -> str:
    return f"Question: {question}\n\n{STYLE_INSTRUCTIONS['short']}"


def image_b64(imgname: str) -> str:
    """The chart PNG as base64 text, the format the model APIs expect."""
    return base64.b64encode((TEST_DIR / "png" / imgname).read_bytes()).decode("ascii")


# ===================================================================== 2. model output -> table
def _markdown_to_csv(lines):
    """'| a | b |' rows -> 'a,b'; separator rows like '|---|---|' are dropped."""
    out = []
    for ln in lines:
        if re.fullmatch(r"\s*\|?[\s:\-|]+\|?\s*", ln):
            continue
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        out.append(",".join(f'"{c}"' if "," in c else c for c in cells))
    return out


def parse_table_text(text: str):
    """Turn a model's reply into a clean table (index = labels, columns = series).

    Accepts a ```csv block, a markdown table, or bare comma lines, and fixes the
    two layouts small vision models often produce:
      - SIDEWAYS ("wide"): categories across the top, values in one row. It is
        turned round so categories run down the first column.
      - MISSING ROW LABEL: a value row with one cell fewer than the header. Without
        this fix every value would slide one column to the left.
    Returns None if nothing usable is found.
    """
    if not text:
        return None
    m = re.search(r"```(?:csv)?\s*\n(.*?)```", text, re.S)
    body = m.group(1) if m else text
    lines = [ln for ln in body.splitlines() if ln.strip()]
    if sum("|" in ln for ln in lines) >= 2:
        lines = _markdown_to_csv([ln for ln in lines if "|" in ln])
    lines = [ln for ln in lines if "," in ln]
    if len(lines) < 2:
        return None

    rows = [[c.strip() for c in r] for r in csv.reader(lines, skipinitialspace=True)]
    rows = [r for r in rows if any(r)]
    header, data = rows[0], rows[1:]
    if not data:
        return None
    if all(len(r) == len(header) - 1 for r in data):          # value rows lack their label
        data = [[f"series {i + 1}"] + r for i, r in enumerate(data)]
    width = len(header)
    data = [(r + [""] * width)[:width] for r in data]          # pad / trim ragged rows

    # Category x category GRID? Some models repeat the categories as both the rows
    # and the columns. Two styles turn up:
    #   diagonal     : each value sits where a category's row meets its own column
    #   first column : each value sits in the first column, the rest is blank/zero
    # We use whichever of the two actually holds the values (diagonal on a tie).
    row_names, col_names = [_norm(r[0]) for r in data], [_norm(h) for h in header[1:]]
    if len(data) >= 2 and sorted(row_names) == sorted(col_names) and len(set(row_names)) == len(row_names):
        pos = {name: j + 1 for j, name in enumerate(col_names)}
        padded = [(r + [""] * len(header))[:len(header)] for r in data]
        diagonal = [r[pos[_norm(r[0])]] for r in padded]
        first_col = [r[1] for r in padded]
        filled = lambda vals: sum(1 for v in vals if not np.isnan(to_number(v)) and to_number(v) != 0)
        values = first_col if filled(first_col) > filled(diagonal) else diagonal
        raw = pd.DataFrame({"label": [r[0] for r in data], "value": values})
        table = table_from_raw(raw)
        return None if table is None else table[~table.index.duplicated()]

    # Sideways? (one value row, or far more columns than rows) -> turn it round
    n_rows, n_cols = len(data), width - 1
    if n_rows == 1 or (n_cols >= 4 and n_cols >= 2 * n_rows):
        long = {"label": header[1:]}
        for i, r in enumerate(data):
            name = "value" if n_rows == 1 else (r[0] or f"series {i + 1}")
            long[name] = r[1:]
        raw = pd.DataFrame(long)
    else:
        raw = pd.DataFrame(data, columns=[h or f"col {j}" for j, h in enumerate(header)])

    table = table_from_raw(raw)
    if table is not None:
        table = table[~table.index.duplicated()]              # keep the first of any repeated label
    return table


def extract_table(model_cfg: dict, imgname: str, cache) -> dict:
    """Ask one vision model to read one chart. Cached like every other LLM call."""
    rec = ask(model_cfg, EXTRACT_PROMPT, cache, system=EXTRACT_SYSTEM,
              images=[image_b64(imgname)], max_tokens=config.EXTRACT_MAX_TOKENS)
    return {"raw": rec["answer"], "table": parse_table_text(rec["answer"]),
            "error": rec["error"], "latency_s": rec["latency_s"], "cached": rec["cached"]}


# ===================================================================== 3. lining two tables up
def _norm(x) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_label(x).lower()).strip()


def align_labels(a_labels, b_labels, threshold: int = 85) -> dict:
    """Pair each label in A with the same label in B (exact after tidying; years must
    be identical; other text may differ slightly, RapidFuzz >= threshold). One-to-one."""
    pairs, used = {}, set()
    b_norm = {b: _norm(b) for b in b_labels}
    for a in a_labels:
        na = _norm(a)
        exact = [b for b, nb in b_norm.items() if nb == na and b not in used]
        if exact:
            pairs[a] = exact[0]
            used.add(exact[0])
            continue
        if re.search(r"\d", na):                    # years / numeric labels: exact only
            continue
        scored = [(fuzz.token_sort_ratio(na, nb), b) for b, nb in b_norm.items() if b not in used]
        if scored:
            score, best = max(scored)
            if score >= threshold:
                pairs[a] = best
                used.add(best)
    return pairs


def align_columns(a_cols, b_cols) -> dict:
    """Pair data series: by name if possible; otherwise by position when both
    tables have the same number of series (a single-series chart always pairs)."""
    a_cols, b_cols = list(a_cols), list(b_cols)
    pairs = align_labels(a_cols, b_cols, threshold=80)
    if len(pairs) < len(a_cols) and len(a_cols) == len(b_cols):
        pairs = dict(zip(a_cols, b_cols))
    return pairs


def rel_diff(a: float, b: float) -> float:
    if pd.isna(a) or pd.isna(b):
        return np.inf
    if b == 0:
        return 0.0 if a == 0 else np.inf
    return abs(a - b) / abs(b)


def best_orientation(table, ref_labels):
    """A multi-series chart can be written either way round (topics as rows or as
    columns). Return the table, or its transpose, whichever lines up with
    `ref_labels` better (more matching row labels)."""
    if table is None or table.shape[1] < 2:
        return table
    as_is = len(align_labels(ref_labels, table.index))
    flipped = len(align_labels(ref_labels, table.T.index))
    return table.T if flipped > as_is else table


# ===================================================================== 4. scoring a reading against the truth
def score_extraction(pred, true, tol: float = config.AGREE_TOL) -> dict:
    """How well did a model read the chart? Compared cell by cell with the true table."""
    n_true = int(true.notna().sum().sum())
    out = {"parsed": pred is not None, "true_cells": n_true, "cells_found": 0,
           "cells_correct": 0, "label_recall": 0.0}
    if pred is None:
        return out
    pred = best_orientation(pred, true.index)
    rows = align_labels(true.index, pred.index)
    cols = align_columns(true.columns, pred.columns)
    out["label_recall"] = len(rows) / len(true.index)
    for r_true, r_pred in rows.items():
        for c_true, c_pred in cols.items():
            t = true.at[r_true, c_true]
            if pd.isna(t):
                continue
            p = pred.at[r_pred, c_pred]
            if pd.notna(p):
                out["cells_found"] += 1
                out["cells_correct"] += int(rel_diff(p, t) <= tol)
    return out


# ===================================================================== 5. the cross-check
def cross_check(t1, t2, tol: float = config.AGREE_TOL):
    """Compare two readings of the same chart.

    Returns (consensus, uncertain):
      consensus : reading 1 (or reading 2 if 1 failed) - the table the oracle will use
      uncertain : same shape, True where the readings disagree or only one reading
                  has the value. False = 'confirmed' by both models.
    """
    if t1 is None and t2 is None:
        return None, None
    if t1 is None or t2 is None:
        base = t1 if t1 is not None else t2
        return base, pd.DataFrame(True, index=base.index, columns=base.columns)
    t2 = best_orientation(t2, t1.index)                      # line reading 2 up with reading 1
    uncertain = pd.DataFrame(True, index=t1.index, columns=t1.columns)
    rows = align_labels(t1.index, t2.index)
    cols = align_columns(t1.columns, t2.columns)
    for r1, r2 in rows.items():
        for c1, c2 in cols.items():
            if rel_diff(t1.at[r1, c1], t2.at[r2, c2]) <= tol:
                uncertain.at[r1, c1] = False
    return t1, uncertain


def confirmed_correct(consensus, uncertain, true, tol: float = config.AGREE_TOL) -> dict:
    """Of the values the cross-check CONFIRMED, how many are actually right?
    And of those it marked UNCERTAIN? (Measures whether the cross-check is useful.)"""
    out = {"confirmed": 0, "confirmed_right": 0, "uncertain": 0, "uncertain_right": 0}
    if consensus is None:
        return out
    if best_orientation(consensus, true.index) is not consensus:   # reading is sideways
        consensus, uncertain = consensus.T, uncertain.T
    rows = align_labels(consensus.index, true.index)
    cols = align_columns(consensus.columns, true.columns)
    for r, rt in rows.items():
        for c, ct in cols.items():
            v, t = consensus.at[r, c], true.at[rt, ct]
            if pd.isna(v) or pd.isna(t):
                continue
            right = int(rel_diff(v, t) <= tol)
            key = "uncertain" if uncertain.at[r, c] else "confirmed"
            out[key] += 1
            out[f"{key}_right"] += right
    return out


# ===================================================================== 6. which values does an answer depend on?
def cells_used(question: str, qtype: str, table):
    """The (row, column) cells the oracle's answer depends on, mirroring oracle():
    a lookup uses one cell; sum/avg/diff/compare of named rows use those rows;
    max/min/trend (and 'all') use the whole series. None if it can't be worked out."""
    if table is None:
        return None
    if qtype == "lookup":
        rows, cols = mentioned(table.index, question), mentioned(table.columns, question)
        if table.shape[1] > 1 and len(rows) == 1 and len(cols) == 1:
            return [(rows[0], cols[0])]
        if table.shape[1] == 1 and len(rows) == 1:
            return [(rows[0], table.columns[0])]
        return None
    s = pick_series(table, question)
    if s is None:
        return None
    named = [r for r in mentioned(s.index, question) if r in s.index]
    if qtype in ("sum", "avg", "diff", "compare") and len(named) >= 2:
        return [(r, s.name) for r in named]
    return [(r, s.name) for r in s.index]


def same_labels(t1, t2) -> bool:
    """True if both readings found exactly the same categories/years. If one reading
    dropped a bar, a 'highest'/'lowest' answer can't be trusted even when every value
    present was confirmed."""
    if t1 is None or t2 is None or len(t1.index) != len(t2.index):
        return False
    return len(align_labels(t1.index, t2.index)) == len(t1.index)


def answer_confidence(question, qtype, consensus, uncertain, labels_match: bool) -> str:
    """'verified' if every value the answer needs was confirmed by both models (and,
    for whole-series questions like max/min/trend, both found the same categories);
    otherwise 'estimated'."""
    cells = cells_used(question, qtype, consensus)
    if not cells:
        return "estimated"
    whole_series = qtype != "lookup" and len(cells) == consensus[cells[0][1]].notna().sum()
    if whole_series and not labels_match:
        return "estimated"
    return "estimated" if any(uncertain.at[r, c] for r, c in cells) else "verified"


def oracle_on_reading(question, qtype, table, uncertain=None):
    """Run the oracle on an extracted table. If it can't answer (often because the
    model wrote a multi-series chart sideways), try the table the other way round.
    Returns (result, reason, table_used, uncertain_used)."""
    if table is None:
        return None, "no_reading", None, None
    result, reason = oracle(question, qtype, table)
    if result is None and table.shape[1] >= 2:
        flipped, why = oracle(question, qtype, table.T)
        if flipped is not None:
            return flipped, why, table.T, (None if uncertain is None else uncertain.T)
    return result, reason, table, uncertain
