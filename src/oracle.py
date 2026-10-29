"""
oracle.py — question typing and the PROGRAMMATIC ORACLE.

The oracle computes the true answer to a question straight from the chart's
data table using pandas. It is plain, deterministic code: the same input
always gives the same output, and no LLM is ever involved.

Golden rule: the oracle REFUSES rather than GUESSES. If a question is
ambiguous it returns None plus a reason, and that question is excluded.
A wrong oracle answer would poison every result downstream.
"""
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .config import FUZZ_THRESHOLD, REL_TOL
from .data_utils import clean_label, looks_like_time, to_number


# ===================================================================== 1. question typing
# Keyword "families". A question that uses more than one family
# ("is the SUM of the two smallest GREATER than ...") is multi-step,
# so we skip it rather than risk a wrong answer.
FAMILIES = {
    "sum":   r"\bsum\b|\btotal of\b|\bcombined\b|\badd(ed)? up\b|\baltogether\b",
    "diff":  r"\bdifference\b|\bgap\b|\bminus\b",
    "avg":   r"\baverage\b|\bmean\b",
    "ratio": r"\bratio\b|\btimes\b",
    "cmp":   r"\b(greater|more|higher|less|lower|smaller|larger|bigger)\s+th[ae]n\b",
    "max":   r"\b(highest|largest|maximum|most(?! of)|greatest|biggest|peak|tallest|longest)\b",
    "min":   r"\b(lowest|smallest|minimum|least|fewest|shortest)\b",
}

SKIP_RULES = [   # checked first; any match -> question is excluded, with this reason
    ("skip_color",   r"\b(red|blue|green|yellow|orange|grey|gray|black|brown|purple|pink|"
                     r"navy|violet|dark|light|colou?r(ed)?)\b"),
    ("skip_count",   r"\bhow many (bars|colou?rs|segments|categories|lines|data points|years|"
                     r"countries|values|items|food|types|kinds|groups|regions|times)\b"),
    ("skip_complex", r"\b(2nd|second|3rd|third|4th|fourth|5th|fifth|median|mode|top|bottom|half|twice|"
                     r"one third|below|above|under|exceed\w*|over \d|three|two \w+est)\b"
                     r"|\b(less|more) th[ae]n \d"),
]

YESNO_START = re.compile(r"^\s*(is|are|does|do|did|can|was|were|has|have|will)\b", re.I)
TIME_WORDS  = re.compile(r"\b(year|when|since|month|period|season|decade)\b", re.I)
TREND_WORDS = re.compile(r"increas|decreas|declin|\brise\b|\brose\b|\bgr[eo]w|\bdrop|\bfell\b|"
                         r"\bfall\b|upward|downward|\btrend\b", re.I)


def classify_question(q: str) -> str:
    """Give each question exactly one type (or a skip reason)."""
    ql = q.lower()
    for name, pat in SKIP_RULES:
        if re.search(pat, ql):
            return name

    fams = {f for f, pat in FAMILIES.items() if re.search(pat, ql)}

    # "difference between the highest and lowest" is one well-defined operation
    if fams == {"diff", "max", "min"}:
        return "diff"
    if len(fams) > 1 and not fams <= {"cmp", "ratio"}:
        return "skip_complex"

    if YESNO_START.search(q):                       # Yes/No questions
        return "compare" if "cmp" in fams else "skip_yesno"
    if "ratio" in fams:
        return "compare"
    if "diff" in fams:
        return "diff"
    # "average ticket price" is a measure's NAME (a lookup), not an operation
    if "avg" in fams:
        return "avg" if re.search(r"\b(average|mean)\s+(value\s+|percentage\s+|number\s+)?of\b", ql) else "lookup"
    if "sum" in fams:
        return "sum"
    if TREND_WORDS.search(ql) and TIME_WORDS.search(ql):
        return "trend"
    if "max" in fams:
        return "max"
    if "min" in fams:
        return "min"
    return "lookup"


# ===================================================================== 2. which entities?
def mentioned(labels, question: str) -> list:
    """Return the table labels (rows or columns) that the question mentions.

    - Number-like labels (2019, '2019/20') must match EXACTLY as a whole token,
      because fuzzy matching would treat 2019 and 2010 as near-identical.
    - Text labels use RapidFuzz partial_ratio >= FUZZ_THRESHOLD, which forgives
      small spelling differences and extra words around the label.
    - If one hit sits inside another ('Asia' in 'South Asia'), keep the longer.
    """
    ql = question.lower()
    hits = []
    for lab in labels:
        l = str(lab).lower().strip()
        if not l:
            continue
        if re.fullmatch(r"[\d.\-/' q]+", l):                       # number-like label
            if re.search(rf"(?<![\d.]){re.escape(l)}(?![\d.])", ql):
                hits.append(lab)
        elif len(l) >= 3 and (l in ql or fuzz.partial_ratio(l, ql) >= FUZZ_THRESHOLD):
            hits.append(lab)
    return [h for h in hits
            if not any(h != o and str(h).lower() in str(o).lower() for o in hits)]


def pick_series(table: pd.DataFrame, question: str):
    """The data column to use: the one the question names, or the only one."""
    cols = mentioned(table.columns, question)
    if len(cols) == 1:
        return table[cols[0]].dropna()
    if table.shape[1] == 1:
        return table.iloc[:, 0].dropna()
    return None                                  # several series, question doesn't say which


# ===================================================================== 3. answer kind
LABEL_CUE = re.compile(
    r"^\s*(which|who|whom|whose|where|when|since when|in what|in which|during which|from what|from which|what year)\b"
    r"|^\s*what (?!is\b|was\b|'s\b|’s\b|are\b|were\b)\w+"            # 'what region', 'what platform'
    r"|\b(name|represent\w*|category|answer|option|group)\b", re.I)
VALUE_CUE = re.compile(
    r"\b(value|how much|how many|percentage|percent|share|number|amount|rate|price|"
    r"revenue|sales|figure|score|level|count|total)\b|%", re.I)


def wants_label(q: str) -> bool:
    """Does the question want a name/year back ('Which country?') rather than a number?"""
    if LABEL_CUE.search(q):
        return True
    return not VALUE_CUE.search(q)               # no value word -> it's asking for a name


# ===================================================================== 4. the oracle
def _num(v):  return {"answer": round(float(v), 4), "answer_kind": "numeric"}
def _lab(v):  return {"answer": str(v), "answer_kind": "categorical"}
def _yn(b):   return {"answer": "Yes" if b else "No", "answer_kind": "yesno"}


def oracle(question: str, qtype: str, table):
    """Compute the true answer.  Returns (result, None) or (None, skip_reason)."""
    if table is None:
        return None, "no_table"
    if qtype.startswith("skip"):
        return None, qtype
    ql = question.lower()

    # ---------- lookup: "What was X in 2019?" ----------
    if qtype == "lookup":
        rows, cols = mentioned(table.index, question), mentioned(table.columns, question)
        if table.shape[1] > 1 and len(rows) == 1 and len(cols) == 1:
            v = table.loc[rows[0], cols[0]]
        elif table.shape[1] == 1 and len(rows) == 1:
            v = table.loc[rows[0]].iloc[0] if isinstance(table.loc[rows[0]], pd.Series) else np.nan
        else:
            return None, "lookup_ambiguous"
        if isinstance(v, pd.Series):
            return None, "duplicate_label"
        return (_num(v), None) if pd.notna(v) else (None, "missing_value")

    s = pick_series(table, question)
    if s is None or s.empty:
        return None, "series_ambiguous"
    rows = [r for r in mentioned(s.index, question) if r in s.index]
    if s.index.duplicated().any():
        return None, "duplicate_label"

    # ---------- max / min ----------
    if qtype in ("max", "min"):
        if rows:      # names a specific row ("lowest ... in 2010") -> really a lookup; too risky
            return None, "maxmin_names_entity"
        target = s.max() if qtype == "max" else s.min()
        if (s == target).sum() > 1:
            return None, "tie"
        idx = s.idxmax() if qtype == "max" else s.idxmin()
        return (_lab(idx), None) if wants_label(question) else (_num(target), None)

    # ---------- diff ----------
    if qtype == "diff":
        if re.search(FAMILIES["max"], ql) and re.search(FAMILIES["min"], ql):
            return _num(s.max() - s.min()), None
        if len(rows) == 2:
            return _num(abs(s[rows[0]] - s[rows[1]])), None
        return None, "diff_needs_two_entities"

    # ---------- sum / avg ----------
    if qtype in ("sum", "avg"):
        f = np.sum if qtype == "sum" else np.mean
        if len(rows) >= 2:
            return _num(f(s[rows])), None
        if not rows and re.search(r"\ball\b", ql):
            return _num(f(s)), None
        return None, f"{qtype}_ambiguous"

    # ---------- compare: ratio, or "is A greater than B?" ----------
    if qtype == "compare":
        if len(rows) != 2:
            return None, "compare_needs_two_entities"
        # order A, B by where they appear in the QUESTION, not in the table
        a_lab, b_lab = sorted(rows, key=lambda r: ql.find(str(r).lower()))
        a, b = s[a_lab], s[b_lab]
        if re.search(FAMILIES["ratio"], ql):
            return (_num(a / b), None) if b != 0 else (None, "divide_by_zero")
        bigger = re.search(r"greater|more|higher|larger|bigger", ql)
        return _yn(a > b if bigger else a < b), None

    # ---------- trend: "Which year saw the biggest increase / sharpest drop?" ----------
    if qtype == "trend":
        if not looks_like_time(s.index) or len(s) < 3:
            return None, "trend_not_time_series"
        # Only this one well-defined form is handled. Vaguer wordings ('since when',
        # 'began to grow', 'increase again') have no single correct rule, so we skip them.
        if not re.search(r"\b(biggest|largest|sharpest|steepest|drastic|highest|greatest|most)\b", ql):
            return None, "trend_vague"
        s = s.sort_index(key=lambda ix: ix.map(to_number))     # oldest -> newest
        step = s.diff().dropna()                                # change from the previous year
        down = re.search(r"decreas|declin|drop|fell|fall|downward", ql)
        # ChartQA convention: report the year the change ENDS in (2018 for a 2017->2018 jump)
        return _lab(step.idxmin() if down else step.idxmax()), None

    return None, "unhandled_type"


# ===================================================================== 5. gold check
def matches_gold(result: dict, gold: str) -> bool:
    """ChartQA 'relaxed accuracy': numbers within 5%, labels equal after cleaning
    (or RapidFuzz ratio >= threshold to forgive tiny spelling differences)."""
    ans, kind = result["answer"], result["answer_kind"]
    if kind == "numeric":
        g = to_number(gold)
        if np.isnan(g):
            return False
        return abs(ans - g) <= REL_TOL * abs(g) if g != 0 else abs(ans) < 1e-9
    a, g = clean_label(ans).lower(), clean_label(gold).lower()
    return a == g or fuzz.ratio(a, g) >= FUZZ_THRESHOLD
