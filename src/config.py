import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MANIFEST_DIR = DATA_DIR / "manifests"
CHROMA_DIR = DATA_DIR / "chroma"

for _d in (DATA_DIR, MANIFEST_DIR, CHROMA_DIR):
    _d.mkdir(parents=True, exist_ok=True)

CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-5")
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

WINDOW_CHAR_BUDGET = int(os.environ.get("WINDOW_CHAR_BUDGET", "12000"))
WINDOW_OVERLAP_SEGMENTS = int(os.environ.get("WINDOW_OVERLAP_SEGMENTS", "5"))

# Anthropic first-party API pricing, $ per 1M tokens (input, output).
# Used only to estimate ingestion cost for the user before they scale up to bigger playlists.
COST_PER_MTOK = {
    "claude-opus-5": (5.0, 25.0),
    "claude-opus-5-5": (4.0, 20.0),
    "claude-sonnet-5": (2.0, 10.0),
    "claude-haiku-4-5": (1.0, 5.0),
}


def estimate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    input_rate, output_rate = COST_PER_MTOK.get(model, COST_PER_MTOK["claude-opus-5"])
    return (input_tokens / 1_000_000) * input_rate + (output_tokens / 1_000_000) * output_rate
