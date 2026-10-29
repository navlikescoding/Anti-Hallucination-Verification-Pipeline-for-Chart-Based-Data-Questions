"""
config.py — every setting the project uses, in one place.

Change a value here and every notebook picks it up. Nothing secret lives here:
the API key is read from an environment variable in Section 2.
"""
import random
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------- paths
ROOT       = Path(__file__).resolve().parent.parent      # the project folder
DATA_DIR   = ROOT / "data"
CACHE_DIR  = ROOT / "cache"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR    = OUTPUT_DIR / "figures"
TABLE_OUT  = OUTPUT_DIR / "tables"
LOG_DIR    = OUTPUT_DIR / "logs"

REPO_URL   = "https://github.com/vis-nlp/ChartQA.git"
REPO_DIR   = DATA_DIR / "ChartQA"
TEST_DIR   = REPO_DIR / "ChartQA Dataset" / "test"
TABLE_DIR  = TEST_DIR / "tables"

# ---------------------------------------------------------------- reproducibility
SEED = 42

def set_seeds(seed: int = SEED) -> None:
    """Fix every random number generator we use, so reruns give identical results."""
    random.seed(seed)
    np.random.seed(seed)

# ---------------------------------------------------------------- matching
REL_TOL        = 0.05   # numbers within 5% count as equal (ChartQA's "relaxed accuracy")
FUZZ_THRESHOLD = 90     # RapidFuzz score (0-100) needed to call two labels the same

# ---------------------------------------------------------------- sampling
N_TARGET     = 200      # questions sent to the LLM (100-300 keeps cost low)
PER_TYPE_CAP = 45       # max questions per type, so rare types aren't swamped


def make_dirs() -> None:
    for d in (DATA_DIR, CACHE_DIR, OUTPUT_DIR, FIG_DIR, TABLE_OUT, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- LLMs (Section 2)
# Each entry is one model we test. 'name' is our short label in results files.
MODELS = [
    {"name": "llama3.2-3b",  "provider": "ollama", "model": "llama3.2:3b"},            # open-source, local
    {"name": "gemini-flash", "provider": "gemini", "model": "gemini-3.5-flash-lite"},  # commercial, free tier
]
ANSWER_STYLES = ["short", "sentence"]

OLLAMA_URL      = "http://localhost:11434"
GEMINI_URL      = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_KEY_ENV  = "GEMINI_API_KEY"   # the key is read from this environment variable, never typed in code
GEMINI_DELAY_S  = 4.0                # pause between Gemini calls to stay under free-tier per-minute limits
TEMPERATURE     = 0.0                # 0 = most deterministic answers (reproducible)
MAX_OUTPUT_TOKENS = 200
LLM_CACHE_FILE  = CACHE_DIR / "llm_responses.jsonl"


# ---------------------------------------------------------------- chart images (Section 3b)
# Two DIFFERENT small vision models read each chart, so their mistakes are less
# likely to coincide. The first one also answers the questions from the image.
VISION_MODELS = [
    {"name": "qwen2.5vl-3b", "provider": "ollama", "model": "qwen2.5vl:3b"},
    {"name": "gemma3-4b",    "provider": "ollama", "model": "gemma3:4b"},
]
N_IMAGE_CHARTS     = 40      # charts to test (each costs ~3 vision calls; ~1-3 h on a CPU)
AGREE_TOL          = 0.05    # two readings of a value "agree" if within 5% of each other
EXTRACT_MAX_TOKENS = 1500    # a table needs more room than a short answer
