"""Reads a video's FULL transcript in one pass (windowed for long transcripts) and
extracts every point made, each tagged core/supporting/minor with a timestamp. This
checklist is the completeness guarantee - it is not just RAG chat.

Concepts are extracted by *transcript segment index*, not free-form timestamps: each
window shows the model numbered lines ("[37] ..."), and the model references a
start_index/end_index range rather than inventing a time. The real timestamp is looked
up from the segment index afterward, in code. Indices are global across the whole
video (not reset per window), so there is no timestamp drift between windows - the
failure mode this is designed to avoid.

Structured output (Pydantic `output_format`) makes the API itself return schema-valid
JSON, which is what keeps "malformed JSON from the LLM" from being a real failure mode
here; the retry-then-skip-this-window behavior below is only a backstop for outright
API errors or a response that fails Pydantic validation.
"""
import difflib
import json
import logging
import time
from dataclasses import asdict, dataclass
from typing import List, Literal, Optional, Tuple

import anthropic
from pydantic import BaseModel, ValidationError

from src import config
from src.fetch_transcripts import Transcript, TranscriptSegment

logger = logging.getLogger(__name__)

Tag = Literal["core", "supporting", "minor"]


class ExtractedConcept(BaseModel):
    start_index: int
    end_index: int
    tag: Tag
    point: str


class ExtractionResult(BaseModel):
    concepts: List[ExtractedConcept]


@dataclass
class Concept:
    id: int
    tag: Tag
    point: str
    start_seconds: float
    end_seconds: float


@dataclass
class IngestStats:
    windows: int
    windows_failed: int
    input_tokens: int
    output_tokens: int
    elapsed_seconds: float


SYSTEM_PROMPT = (
    "You extract a completeness checklist from a lecture/video transcript excerpt. "
    "Read every line and list EVERY distinct point the speaker makes in this excerpt - "
    "do not summarize or skip minor asides. Tag each point:\n"
    "- core: a main idea the video is built around\n"
    "- supporting: elaborates, justifies, or gives an example of a core idea\n"
    "- minor: an aside, joke, announcement, or minor detail\n"
    "Reference each point by the transcript line index range it comes from "
    "(start_index/end_index, inclusive, matching the [N] markers in the transcript). "
    "Only use indices that appear in the excerpt shown to you. If a point spans "
    "multiple lines, give the full range it spans."
)


def _windows(
    segments: List[TranscriptSegment], char_budget: int, overlap: int
) -> List[List[TranscriptSegment]]:
    windows: List[List[TranscriptSegment]] = []
    window: List[TranscriptSegment] = []
    chars = 0
    for seg in segments:
        window.append(seg)
        chars += len(seg.text)
        if chars >= char_budget:
            windows.append(window)
            window = window[-overlap:] if overlap else []
            chars = sum(len(s.text) for s in window)
    if window:
        windows.append(window)
    return windows


def _render_window(segments: List[TranscriptSegment]) -> str:
    return "\n".join(f"[{s.index}] {s.text}" for s in segments)


def _clamp_to_window(concept: ExtractedConcept, min_idx: int, max_idx: int) -> ExtractedConcept:
    if concept.start_index < min_idx or concept.end_index > max_idx:
        concept.start_index = max(min_idx, min(concept.start_index, max_idx))
        concept.end_index = max(concept.start_index, min(concept.end_index, max_idx))
    return concept


def _dedupe(concepts: List[ExtractedConcept]) -> List[ExtractedConcept]:
    kept: List[ExtractedConcept] = []
    for c in concepts:
        is_dupe = False
        for k in kept:
            overlap = min(c.end_index, k.end_index) - max(c.start_index, k.start_index)
            if overlap >= 0 and difflib.SequenceMatcher(None, c.point, k.point).ratio() > 0.6:
                is_dupe = True
                break
        if not is_dupe:
            kept.append(c)
    return kept


def _extract_window(
    client: anthropic.Anthropic, window: List[TranscriptSegment]
) -> Tuple[Optional[ExtractionResult], int, int]:
    min_idx, max_idx = window[0].index, window[-1].index
    prompt = (
        f"Transcript excerpt, lines [{min_idx}]-[{max_idx}]. Only extract points that "
        f"start within this range (earlier context may repeat prior excerpts and should "
        f"not be re-extracted unless a new point only becomes clear here):\n\n"
        f"{_render_window(window)}"
    )

    last_error: Optional[Exception] = None
    for attempt in range(2):
        try:
            response = client.messages.parse(
                model=config.CLAUDE_MODEL,
                max_tokens=8000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_format=ExtractionResult,
            )
            return response.parsed_output, response.usage.input_tokens, response.usage.output_tokens
        except (anthropic.APIError, ValidationError, ValueError) as e:
            last_error = e
            logger.warning(
                "Extraction window [%d-%d] attempt %d/2 failed: %s", min_idx, max_idx, attempt + 1, e
            )

    logger.error("Giving up on window [%d-%d] after retries: %s", min_idx, max_idx, last_error)
    return None, 0, 0


def extract_concepts(
    transcript: Transcript,
    client: Optional[anthropic.Anthropic] = None,
    window_char_budget: Optional[int] = None,
    window_overlap: Optional[int] = None,
) -> Tuple[List[Concept], IngestStats]:
    client = client or anthropic.Anthropic()
    window_char_budget = window_char_budget or config.WINDOW_CHAR_BUDGET
    window_overlap = config.WINDOW_OVERLAP_SEGMENTS if window_overlap is None else window_overlap

    segments_by_index = {s.index: s for s in transcript.segments}
    all_concepts: List[ExtractedConcept] = []

    windows_seen = 0
    windows_failed = 0
    input_tokens = 0
    output_tokens = 0
    started = time.monotonic()

    for window in _windows(transcript.segments, window_char_budget, window_overlap):
        windows_seen += 1
        min_idx, max_idx = window[0].index, window[-1].index
        parsed, in_tok, out_tok = _extract_window(client, window)
        input_tokens += in_tok
        output_tokens += out_tok

        if parsed is None:
            windows_failed += 1
            continue

        for concept in parsed.concepts:
            all_concepts.append(_clamp_to_window(concept, min_idx, max_idx))

    deduped = _dedupe(all_concepts)

    concepts: List[Concept] = []
    for i, c in enumerate(deduped):
        start_seg = segments_by_index.get(c.start_index)
        end_seg = segments_by_index.get(c.end_index) or start_seg
        if start_seg is None:
            continue
        concepts.append(
            Concept(
                id=i,
                tag=c.tag,
                point=c.point,
                start_seconds=start_seg.start,
                end_seconds=end_seg.start + end_seg.duration,
            )
        )

    stats = IngestStats(
        windows=windows_seen,
        windows_failed=windows_failed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        elapsed_seconds=time.monotonic() - started,
    )
    return concepts, stats


def save_manifest(video_id: str, concepts: List[Concept]) -> None:
    path = config.MANIFEST_DIR / f"{video_id}.json"
    path.write_text(json.dumps([asdict(c) for c in concepts], indent=2), encoding="utf-8")


def load_manifest(video_id: str) -> List[Concept]:
    path = config.MANIFEST_DIR / f"{video_id}.json"
    if not path.exists():
        return []
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Concept(**item) for item in raw]
