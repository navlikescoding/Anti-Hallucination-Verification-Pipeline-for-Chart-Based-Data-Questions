"""
data_utils.py — downloading ChartQA, loading Q&A pairs, cleaning tables,
and guessing each chart's type.
"""
import json
import logging
import re
import subprocess

import numpy as np
import pandas as pd

from .config import REPO_DIR, REPO_URL, TABLE_DIR, TEST_DIR

log = logging.getLogger("pipeline")


# ===================================================================== download
def download_chartqa() -> None:
    """Fetch only the test split's tables + Q&A JSON (a few MB, no images).

    Uses git's 'sparse checkout' so we don't download the whole repo.
    If git is missing, download the repo ZIP from GitHub and unzip it into data/ChartQA.
    """
    if (TEST_DIR / "test_human.json").exists():
        print("ChartQA already downloaded.")
        return
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse",
                    REPO_URL, str(REPO_DIR)], check=True)
    subprocess.run(["git", "-C", str(REPO_DIR), "sparse-checkout", "set", "--no-cone",
                    "ChartQA Dataset/test/*.json", "ChartQA Dataset/test/tables/*"], check=True)
    print("Downloaded ChartQA test split.")


def download_images(imgnames) -> None:
    """Fetch only the listed chart images (PNG) from ChartQA's test split (Section 3b)."""
    missing = [n for n in imgnames if not (TEST_DIR / "png" / n).exists()]
    if not missing:
        return
    patterns = [f"/ChartQA Dataset/test/png/{n}" for n in missing]
    for i in range(0, len(patterns), 100):                   # a few at a time keeps the command short
        subprocess.run(["git", "-C", str(REPO_DIR), "sparse-checkout", "add", *patterns[i:i + 100]],
                       check=True, capture_output=True)
    print(f"Downloaded {len(missing)} chart images.")


# ===================================================================== Q&A pairs
def load_qa() -> pd.DataFrame:
    """Load human + augmented test questions into one table.

    Columns: qid, source, imgname, table_file, query, label (the gold answer).
    """
    parts = []
    for source in ("human", "augmented"):
        with open(TEST_DIR / f"test_{source}.json", encoding="utf-8") as f:
            df = pd.DataFrame(json.load(f))
        df["source"] = source
        parts.append(df)
    qa = pd.concat(parts, ignore_index=True)
    qa["table_file"] = qa["imgname"].str.replace(".png", ".csv", regex=False)
    qa["label"] = qa["label"].astype(str).str.strip()
    qa.insert(0, "qid", range(len(qa)))          # stable ID for every question
    return qa


# ===================================================================== cleaning
_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

def to_number(x) -> float:
    """Turn a messy cell into a float.  '36%' -> 36.0, '$1,200' -> 1200.0,
    '457*' -> 457.0, '-' or empty -> NaN."""
    if pd.isna(x):
        return np.nan
    s = str(x).replace(",", "").replace("−", "-").strip()   # − = unicode minus sign
    m = _NUM_RE.search(s)
    return float(m.group()) if m else np.nan


def clean_label(x) -> str:
    """Remove footnote stars and extra spaces:  '2021*' -> '2021'."""
    return re.sub(r"\s+", " ", str(x).replace("*", "")).strip()


def load_table(table_file: str):
    """Read one chart's CSV and return a clean numeric table, or None if unusable.

    Convention: the FIRST column holds labels (categories or years) and becomes
    the index; every other column is one data series.
    """
    try:
        raw = pd.read_csv(TABLE_DIR / table_file, dtype=str)
    except Exception as e:                                   # missing / corrupt file
        log.info(f"table_unreadable | {table_file} | {e}")
        return None
    if raw.shape[1] < 2 or raw.shape[0] < 1:
        log.info(f"table_too_small | {table_file}")
        return None
    data = table_from_raw(raw)
    if data is None:
        log.info(f"table_no_numeric_data | {table_file}")
    return data


def table_from_raw(raw: pd.DataFrame):
    """Clean a raw table (all text): first column -> labels (index), every other
    column -> numbers. Shared by load_table and the chart-image extractor (3b)."""
    if raw.shape[1] < 2 or raw.shape[0] < 1:
        return None
    data = raw.iloc[:, 1:].apply(lambda col: col.map(to_number))
    data.index = raw.iloc[:, 0].map(clean_label)
    data.columns = [clean_label(c) for c in data.columns]
    data = data.dropna(axis=1, how="all")                    # drop columns with no numbers
    return None if data.empty else data


# ===================================================================== chart type
_TIME_RE = re.compile(
    r"^(?:Q[1-4]\s*)?'?\d{2,4}(?:[/\-]\d{2,4})?(?:\s*Q[1-4])?$"    # 2019, 2019/20, Q1 2020
    r"|^[A-Z][a-z]{2,8}\.?\s+'?\d{2,4}$")                        # Jan 2021, March '20


def looks_like_time(labels) -> bool:
    """True if at least 80% of the labels look like years or dates."""
    labels = list(labels)
    hits = sum(bool(_TIME_RE.match(str(l))) for l in labels)
    return len(labels) > 0 and hits >= 0.8 * len(labels)


def infer_chart_type(table):
    """HEURISTIC — ChartQA's tables don't record the chart type, so we guess:
       line : labels are years/dates and there are >= 5 points
       pie  : one series, <= 8 slices, values add up to ~100
       bar  : everything else
    """
    if table is None:
        return None
    if looks_like_time(table.index) and len(table) >= 5:
        return "line"
    if table.shape[1] == 1 and len(table) <= 8 and 95 <= table.iloc[:, 0].sum() <= 105:
        return "pie"
    return "bar"
