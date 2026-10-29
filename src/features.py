"""
features.py — features for the hallucination classifier (Section 4).

THE RULE: every feature here must be known at the moment the LLM answers,
WITHOUT the true answer. So nothing from the oracle or the verifier is used:
no numeric deviation, no fuzzy score against the truth, no cosine similarity
against a reference answer. (Those would make the classifier circular - it would
just relearn the verifier's own thresholds.)

What IS allowed: who answered (model, style), what was asked (question type,
length), what the chart looks like (chart type, table size), and what the answer
looks like (length, numbers, hedging, refusal wording, whether it names a
category from the chart).
"""
import re

import numpy as np
import pandas as pd

from .data_utils import load_table
from .oracle import wants_label
from .verify import REFUSAL_RE, numbers_in, find_labels

HEDGE_RE = re.compile(r"\b(approximately|about|around|roughly|estimated|nearly|almost|"
                      r"approx|close to|just over|just under)\b", re.I)

CATEGORICAL = ["model", "style", "qtype", "chart_type"]
NUMERIC = ["answer_words", "question_words", "answer_numbers", "is_refusal", "is_hedged",
           "names_a_category", "number_only_for_name_question", "table_rows", "table_cols"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """One row of features per LLM answer, using only information available
    before the answer is checked."""
    tables = {f: load_table(f) for f in df["table_file"].unique()}
    rows = []
    for r in df.itertuples():
        ans = "" if pd.isna(r.llm_answer) else str(r.llm_answer)
        table = tables.get(r.table_file)
        labels = list(table.index) if table is not None else []
        asks_name = wants_label(r.query)
        names_cat = bool(find_labels(labels, ans)) if ans else False
        rows.append({
            "model": r.model,
            "style": r.style,
            "qtype": r.qtype,
            "chart_type": r.chart_type,
            "answer_words": len(ans.split()),
            "question_words": len(str(r.query).split()),
            "answer_numbers": len(numbers_in(ans)),
            "is_refusal": int(bool(REFUSAL_RE.search(ans))),
            "is_hedged": int(bool(HEDGE_RE.search(ans))),
            "names_a_category": int(names_cat),
            # asked "which X?" but the answer contains a number and no category name
            "number_only_for_name_question": int(asks_name and not names_cat and bool(numbers_in(ans))),
            "table_rows": 0 if table is None else table.shape[0],
            "table_cols": 0 if table is None else table.shape[1],
        })
    return pd.DataFrame(rows, index=df.index)
