"""
verify.py — THE VERIFICATION LAYER.

For each LLM answer it:
  1. extracts what the LLM actually claimed (a number, a name, or Yes/No),
     from either a short answer or a full sentence,
  2. compares that claim with the oracle's true answer,
  3. returns a status (correct / wrong_value / wrong_entity / ...) plus the
     numeric features the classifier in Section 4 will use.

Everything here is deterministic code. No LLM is ever used to judge an answer.
"""
import re

import numpy as np
import pandas as pd
from rapidfuzz import fuzz

from .config import FUZZ_THRESHOLD, REL_TOL
from .data_utils import clean_label, to_number

LOOKUP_TOL = 0.005    # lookups copy a value from the table, so they must be (almost) exact

STATUSES = ["correct", "wrong_value", "wrong_entity", "wrong_answer_type",
            "refusal", "unparseable", "no_answer"]

# Phrases where the model claims the data isn't there (it always is, in our sample)
REFUSAL_RE = re.compile(
    r"not available|no data|not provided|not included|does not include|do not have|"
    r"don't have|cannot (be )?determine|can't determine|unable to|not specified|"
    r"insufficient|not possible to|no information|\bn/a\b", re.I)

_NUM_RE   = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")
_RATIO_RE = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\s*:\s*(-?\d[\d,]*(?:\.\d+)?)")


# ===================================================================== 1. extraction helpers
_RATIO_WORDS_RE = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)%?\s+to\s+(-?\d[\d,]*(?:\.\d+)?)")
_ZERO_RE  = re.compile(r"\bno difference\b|\bthe same\b|\bequal\b|\bidentical\b", re.I)
_RANGE_RE = re.compile(r"\b\d+\s*(?:-|–|—|to)\s*\d+\s*(?=years|yrs|year-olds|age|y/o)|"   # 18 to 29 years
                       r"\b\d+\s*[-–—]\s*\d+\b|\b\d+\s*\+")                              # 25-34, 65+

# ---- scale words (fix 5): '77.5 billion' vs a table 'in million euros'
SCALE = {"thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12}
_SCALED_RE = re.compile(r"(-?\d[\d,]*(?:\.\d+)?)\s*(thousand|million|billion|trillion)\b", re.I)

# ---- number words (fix 5): only when followed by a unit, so 'one of the' is never read as 1
_WORDS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen twenty".split())}
_WORDS.update({"thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
               "eighty": 80, "ninety": 90, "hundred": 100})
_DIGITS = "zero one two three four five six seven eight nine".split()
_WORDNUM_RE = re.compile(r"\b(" + "|".join(_WORDS) + r")(\s+and\s+a\s+half|\s+point\s+(?:"
                         + "|".join(_DIGITS) + r"))?"
                         r"(?=\s*(?:%|percent|thousand|million|billion|trillion))", re.I)


def tidy(text: str) -> str:
    """Clean an answer before reading numbers:
    - 'in 2007.' -> 'in 2007 '   (a full stop after a number is not a decimal point)
    - 'three and a half billion' -> '3.5 billion'"""
    text = re.sub(r"(\d)\.(?!\d)", r"\1 ", str(text))
    def word_to_num(m):
        v = _WORDS[m.group(1).lower()]
        extra = (m.group(2) or "").lower().split()
        if extra[:1] == ["and"]:
            v += 0.5                                         # 'three and a half'
        elif extra[:1] == ["point"]:
            v += _DIGITS.index(extra[1]) / 10                # 'two point five'
        return f"{v:g}"
    return _WORDNUM_RE.sub(word_to_num, text)


def numbers_in(text: str, ratio_words: bool = False) -> list:
    """All numbers in a piece of text. A ratio 'a:b' becomes a/b; with ratio_words=True
    so does 'a to b' (only used for ratio questions, so 'from 2006 to 2016' is safe).
    Phrases like 'no difference' add a 0."""
    text = tidy(text)
    out = [0.0] if _ZERO_RE.search(text) else []
    if ratio_words:
        for m in _RATIO_WORDS_RE.finditer(text):            # '1 to 4' -> 0.25
            a, b = to_number(m.group(1)), to_number(m.group(2))
            if b:
                out.append(a / b)
        text = _RATIO_WORDS_RE.sub(" ", text)
    for m in _RATIO_RE.finditer(text):                      # '1:4' -> 0.25
        a, b = to_number(m.group(1)), to_number(m.group(2))
        if b:
            out.append(a / b)
    text = _RATIO_RE.sub(" ", text)
    for tok in _NUM_RE.findall(text):
        v = to_number(tok)
        if not np.isnan(v):
            out.append(v)
    return out


def value_unit(table, question=""):
    """The unit the table's values are in ('GDP in million euros' -> 1e6), read from the
    column headers, or failing that from the question ('... (in million U.S. dollars)?').
    None if no unit is stated (or the headers disagree): then scale words are ignored."""
    def units_in(texts):
        return {w.lower() for t in texts for w in re.findall(
            r"\b(thousand|million|billion|trillion)s?\b", str(t), re.I)}
    found = units_in(table.columns) if table is not None else set()
    if not found:
        found = units_in([question])
    return SCALE[found.pop()] if len(found) == 1 else None


def tolerance_for(qtype: str) -> float:
    """0.5% for lookups (copy a value), 5% for computed answers (rounding is fair)."""
    return LOOKUP_TOL if qtype == "lookup" else REL_TOL


def rel_deviation(a: float, g: float) -> float:
    """How far a is from g, as a fraction of g (0.1 = 10% off). Capped at 10."""
    if g == 0:
        return 0.0 if a == 0 else 10.0
    return float(min(abs(a - g) / abs(g), 10.0))


# ===================================================================== 2. finding table labels in text
GENERIC_LABELS = {"total", "other", "others", "overall", "all", "average", "mean", "none",
                  "rest", "both", "neither", "yes", "no"}


def _aliases(label) -> list:
    """Other ways an answer may name a label: 'Kit (Nike)' -> 'Kit', 'Nike'."""
    lab = str(label)
    out = [lab]
    m = re.match(r"^(.*?)\s*\((.+)\)\s*$", lab)
    if m:
        out += [m.group(1).strip(), m.group(2).strip()]
    return [a for a in out if len(a) >= 2]


def find_labels(labels, text: str) -> list:
    """Which table labels does `text` mention? Returns [(label, position)] in text order.

    Stricter than the oracle's matcher, because answers are long free text:
      - whole-word matches only ('Niger' is not found inside 'Nigeria'),
      - short labels ('UK', 'US') and generic words ('Total', 'Other') must match with
        the SAME capitals, so 'the total consumption' doesn't count as the row 'Total',
      - RapidFuzz (>= 90) is only a fallback for longer labels when nothing matched
        exactly; it catches spelling differences like 'Re publican' vs 'Republican'.
    """
    hits = {}
    for lab in labels:
        for al in _aliases(lab):
            strict = len(al) <= 2 or al.lower() in GENERIC_LABELS
            pat = rf"(?<![\w.]){re.escape(al)}(?![\w])"
            m = re.search(pat, text, 0 if strict else re.I)
            if m and (lab not in hits or m.start() < hits[lab]):
                hits[lab] = m.start()
    if not hits:
        for lab in labels:
            al = str(lab)
            if len(al) >= 4 and len(text) >= len(al) and al.lower() not in GENERIC_LABELS and \
                    fuzz.partial_ratio(al.lower(), text.lower()) >= FUZZ_THRESHOLD:
                hits[lab] = len(text)                       # position unknown: put last
    # drop a label found inside a longer found label ('Asia' inside 'South Asia')
    keep = {l: p for l, p in hits.items()
            if not any(l != o and str(l).lower() in str(o).lower() for o in hits)}
    return sorted(keep.items(), key=lambda kv: kv[1])


def label_spans_removed(text: str, table) -> str:
    """Blank out table labels (and ranges like '25-34', '65+') so the numbers inside them
    ('2019', '25-34 age group') aren't mistaken for the answer (fix 2)."""
    if table is not None:
        names = [str(x) for x in list(table.index) + list(table.columns)]
        for name in sorted(names, key=len, reverse=True):
            if re.search(r"\d", name) and len(name) >= 2:
                text = re.sub(re.escape(name), " ", text, flags=re.I)
    return _RANGE_RE.sub(" ", text)


# ===================================================================== 3. the three checkers
def check_numeric(answer, oracle, question, qtype, style, table=None):
    """Numbers: is the claimed value within tolerance of the oracle?

    short    -> the LAST number in the answer is the claim ('103.9 + 40.8 = 144.7').
    sentence -> sentences often repeat table values, so we take the number CLOSEST
                to the oracle (a lenient 'contains the right value' rule). To stop that
                rule from being fooled, numbers inside table labels, age ranges and the
                question itself are removed first.
    A number followed by 'million'/'billion' is converted to the table's own unit.
    """
    g = float(oracle)
    text = tidy(answer)
    unit = value_unit(table, question)
    # scaled numbers first: '77.5 billion' in a 'million' table -> 77500
    scaled = []
    if unit is not None:
        for m in _SCALED_RE.finditer(text):
            scaled.append(to_number(m.group(1)) * SCALE[m.group(2).lower()] / unit)
        text = _SCALED_RE.sub(" ", text)

    text = label_spans_removed(text, table)
    nums = scaled + numbers_in(text, ratio_words="ratio" in question.lower())
    q_nums = set(numbers_in(label_spans_removed(tidy(question), table)))
    candidates = [n for n in nums if n not in q_nums]   # numbers the answer adds itself
    if not candidates:
        return None, np.nan
    claim = candidates[-1] if style == "short" else min(candidates, key=lambda n: abs(n - g))
    return claim, rel_deviation(claim, g)


def check_categorical(answer, oracle, question, table, style):
    """Names / years: which table label does the answer assert?

    1. If the whole (short) answer exactly equals a label, that's the claim.
    2. Otherwise find the table labels mentioned in the answer (see find_labels),
       ignore labels the question itself mentions, and take the FIRST one in the
       answer as the claim.
    3. A short answer that is part of exactly one label ('May' -> "May '21") counts
       as that label.
    """
    a = clean_label(tidy(answer)).strip(" .")
    labels = list(table.index) if table is not None else [oracle]
    fuzzy = fuzz.partial_ratio(str(oracle).lower(), a.lower())

    for lab in labels:                                                # step 1: exact
        if any(a.lower() == al.lower() for al in _aliases(lab)):
            return lab, fuzzy

    q_ents = {lab for lab, _ in find_labels(labels, question)}
    # a label that also appears in a column header (e.g. '2012' in 'GDP in chained 2012
    # U.S. dollars') is being used as part of the unit, not as the answer
    headers = " ".join(str(c) for c in table.columns) if table is not None else ""
    in_header = {lab for lab, _ in find_labels(labels, headers)} if headers else set()
    ents = [lab for lab, _ in find_labels(labels, a) if lab not in q_ents | in_header]
    if not ents and style == "short" and len(a) >= 3:
        # a short answer that is part of exactly one label: 'May' for "May '21",
        # 'Europe' for 'Europe and Central Asia'
        inside = [lab for lab in labels if lab not in q_ents and
                  re.search(rf"(?<!\w){re.escape(a)}(?!\w)", str(lab), re.I)]
        if len(inside) == 1:
            ents = inside
    return (ents[0] if ents else None), fuzzy


def previous_period(table, label):
    """For trend questions: the period just before `label` in time order
    (2017 for a 2017->2018 change). Used because 'the year of the biggest increase'
    can fairly be named by either end of the change."""
    if table is None or label not in table.index:
        return None
    order = sorted(table.index, key=to_number)
    i = order.index(label)
    return order[i - 1] if i > 0 else None


def check_yesno(answer, question):
    """Yes/No: a leading Yes/No wins; otherwise read 'is less than' / 'is greater than'."""
    a = str(answer).strip().lower()
    m = re.match(r"^\W*(yes|no)\b", a)
    if m:
        return m.group(1).capitalize()
    asks_bigger = bool(re.search(r"greater|more|higher|larger|bigger", question.lower()))
    says_smaller = re.search(r"\b(less|lower|smaller|fewer)\b|not (more|greater|higher|larger)", a)
    says_bigger = re.search(r"\b(greater|more|higher|larger|bigger)\b", a)
    if says_smaller:
        return "No" if asks_bigger else "Yes"
    if says_bigger:
        return "Yes" if asks_bigger else "No"
    return None


# ===================================================================== 3. one answer -> verdict
def verify_one(llm_answer, oracle, answer_kind, question, qtype, style, table=None) -> dict:
    """Verify a single answer. Returns status, the extracted claim, and features."""
    out = {"status": None, "claim": None, "num_deviation": np.nan, "fuzzy_score": np.nan,
           "is_refusal": False, "answer_len": 0}
    if llm_answer is None or (isinstance(llm_answer, float) and np.isnan(llm_answer)) \
            or not str(llm_answer).strip():
        out["status"] = "no_answer"
        return out
    text = str(llm_answer).strip()
    out["answer_len"] = len(text.split())
    out["is_refusal"] = bool(REFUSAL_RE.search(text))

    if answer_kind == "numeric":
        claim, dev = check_numeric(text, oracle, question, qtype, style, table)
        out["claim"], out["num_deviation"] = claim, dev
        if claim is not None and dev <= tolerance_for(qtype):
            out["status"] = "correct"
        elif claim is not None:
            out["status"] = "wrong_value"
        elif out["is_refusal"]:
            out["status"] = "refusal"
        else:
            # e.g. a name ('Alaska') or a table label ('2016') where a value was asked
            out["status"] = "wrong_answer_type" if re.search(r"[A-Za-z]{3}|\d", text) else "unparseable"

    elif answer_kind == "categorical":
        claim, score = check_categorical(text, oracle, question, table, style)
        out["claim"], out["fuzzy_score"] = claim, score
        accepted = {clean_label(oracle).lower()}
        if qtype == "trend":                                  # either end of the change
            prev = previous_period(table, str(oracle))
            if prev is not None:
                accepted.add(clean_label(prev).lower())
        if claim is not None and clean_label(claim).lower() in accepted:
            out["status"] = "correct"
        elif claim is not None:
            out["status"] = "wrong_entity"
        elif out["is_refusal"]:
            out["status"] = "refusal"
        elif numbers_in(text):
            out["status"] = "wrong_answer_type"          # gave a number when a name was asked
        else:
            out["status"] = "unparseable"

    else:  # yesno
        claim = check_yesno(text, question)
        out["claim"] = claim
        if claim == oracle:
            out["status"] = "correct"
        elif out["is_refusal"]:
            out["status"] = "refusal"
        elif claim is None:
            out["status"] = "unparseable"
        else:
            out["status"] = "wrong_value"
    return out


def verify_frame(df: pd.DataFrame, tables: dict, answer_col: str = "llm_answer") -> pd.DataFrame:
    """Run verify_one over every row. Returns new columns to join onto df."""
    rows = []
    for r in df.itertuples():
        v = verify_one(getattr(r, answer_col), r.oracle_answer, r.answer_kind,
                       r.query, r.qtype, r.style, tables.get(r.table_file))
        rows.append(v)
    return pd.DataFrame(rows, index=df.index)


# ===================================================================== 4. cosine similarity
def reference_sentence(question: str, oracle) -> str:
    """A plain reference statement of the true answer, for cosine similarity."""
    return f"{question} The answer is {oracle}."


def cosine_similarities(answers, references, model_name="all-MiniLM-L6-v2"):
    """Cosine similarity between each answer and its reference sentence (0..1).

    Uses Sentence-Transformers on the CPU. Returns NaNs (with a message) if the
    library or model can't be loaded, so the rest of the pipeline still runs.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device="cpu")
    except Exception as e:
        print(f"Cosine similarity skipped ({e}).")
        return np.full(len(answers), np.nan)
    a = model.encode([str(x) for x in answers], batch_size=64, normalize_embeddings=True,
                     show_progress_bar=False)
    b = model.encode([str(x) for x in references], batch_size=64, normalize_embeddings=True,
                     show_progress_bar=False)
    return (a * b).sum(axis=1)                  # dot product of unit vectors = cosine


# ===================================================================== 5. fallback answer
def fmt_value(oracle, answer_kind) -> str:
    """Readable text for an oracle value: 13779.1 -> '13779.1', 10085207.0 -> '10085207'."""
    if answer_kind != "numeric":
        return str(oracle)
    v = float(oracle)
    return str(int(v)) if v.is_integer() else f"{round(v, 4)}"


def template_answer(oracle, answer_kind) -> str:
    """Last-resort answer built only from the oracle, used if re-prompting fails."""
    if answer_kind == "yesno":
        return f"{oracle}."
    return f"The answer is {fmt_value(oracle, answer_kind)}."