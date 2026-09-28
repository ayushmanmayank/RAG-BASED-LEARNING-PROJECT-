"""Offline dry run: exercises the full ingestion + RAG pipeline with fakes for YouTube
and Claude, so it proves the wiring works end-to-end without any network calls or API
spend. Real-world failure modes (missing transcripts, long-transcript windowing, model
JSON drift, cost/time at scale) still need a real run against a real playlist - this
only proves nothing is broken in transit.
"""
import shutil
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config  # noqa: E402
from src.concept_extractor import extract_concepts, load_manifest, save_manifest  # noqa: E402
from src.fetch_transcripts import Transcript, TranscriptSegment  # noqa: E402
from src.rag_engine import RagEngine, Video  # noqa: E402
from src.vectorstore import VectorStore  # noqa: E402

_FAILURES = []


def _check(condition: bool, label: str) -> bool:
    status = "PASS" if condition else "FAIL"
    print(f"{status}: {label}")
    if not condition:
        _FAILURES.append(label)
    return condition


class FakeMessages:
    """Stands in for client.messages: cycles through a scripted list of parsed
    results/replies so extraction and chat/quiz never touch the real API."""

    def __init__(self, script):
        self._script = script
        self._i = 0

    def parse(self, **kwargs):
        output_format = kwargs["output_format"]
        payload = self._script[self._i % len(self._script)]
        self._i += 1
        parsed = output_format.model_validate(payload)
        usage = SimpleNamespace(input_tokens=100, output_tokens=50)
        return SimpleNamespace(usage=usage, parsed_output=parsed)

    def create(self, **kwargs):
        text_block = SimpleNamespace(type="text", text="Fake chat answer grounded in the excerpts above.")
        return SimpleNamespace(content=[text_block])


class FakeClient:
    def __init__(self, script):
        self.messages = FakeMessages(script)


def make_transcript(video_id: str, n_segments: int, topic: str) -> Transcript:
    segments = [
        TranscriptSegment(index=i, start=float(i * 5), duration=5.0, text=f"{topic} point number {i}.")
        for i in range(n_segments)
    ]
    return Transcript(video_id=video_id, language="en", is_generated=False, segments=segments)


def run() -> bool:
    data_dir = config.DATA_DIR / "_dry_run"
    if data_dir.exists():
        shutil.rmtree(data_dir)
    data_dir.mkdir(parents=True)
    config.MANIFEST_DIR = data_dir / "manifests"
    config.MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    chroma_dir = data_dir / "chroma"

    extraction_script = [
        {"concepts": [{"start_index": 0, "end_index": 1, "tag": "core", "point": "Gradient descent minimizes loss."}]},
        {"concepts": [{"start_index": 2, "end_index": 3, "tag": "supporting", "point": "Learning rate controls step size."}]},
        {"concepts": [{"start_index": 4, "end_index": 5, "tag": "minor", "point": "Aside about notation."}]},
    ]

    # 1. Windowed concept extraction, including out-of-range indices the extractor
    #    must clamp rather than trust, and duplicate points across overlapping windows
    #    the extractor must dedupe.
    transcript_a = make_transcript("vidA", n_segments=6, topic="Gradient descent")
    concepts_a, stats_a = extract_concepts(
        transcript_a,
        client=FakeClient(extraction_script),
        window_char_budget=40,
        window_overlap=1,
    )
    _check(len(concepts_a) >= 1, "extraction produced at least one concept")
    _check(stats_a.windows > 1, "long transcript was split into multiple windows")
    _check(stats_a.windows_failed == 0, "no windows failed against the fake client")
    _check(
        all(0 <= c.start_seconds <= 30 for c in concepts_a),
        "clamped concepts carry timestamps within the video's real duration",
    )

    save_manifest("vidA", concepts_a)
    _check(len(load_manifest("vidA")) == len(concepts_a), "manifest round-trips through disk")

    # 2. A video with zero extracted concepts (e.g. no transcript) must not break ingestion.
    transcript_b = make_transcript("vidB", n_segments=4, topic="Backpropagation")
    concepts_b, _ = extract_concepts(
        transcript_b, client=FakeClient(extraction_script), window_char_budget=40, window_overlap=1
    )
    save_manifest("vidB", concepts_b)

    # 3. Vectorstore chronology scoping: a later video's concepts must not leak backward.
    store = VectorStore(path=chroma_dir)
    store.add_concepts("vidA", playlist_index=0, concepts=concepts_a)
    store.add_concepts("vidB", playlist_index=1, concepts=concepts_b)

    hits_locked = store.search("loss", max_playlist_index=0, n_results=10)
    _check(
        all(h["playlist_index"] <= 0 for h in hits_locked),
        "chronology filter excludes videos later than the current one",
    )

    hits_unlocked = store.search("loss", max_playlist_index=1, n_results=10)
    _check(
        any(h["video_id"] == "vidB" for h in hits_unlocked),
        "later video becomes visible once its index is unlocked",
    )

    # 4. RAG engine chat/summarize/quiz against fakes, with fetch_transcript patched so
    #    the "re-fetch the actual words live" path doesn't hit the real network.
    import src.rag_engine as rag_module

    videos = [Video("vidA", "Video A", 0), Video("vidB", "Video B", 1)]
    chat_client = FakeClient(extraction_script)
    engine = RagEngine(store, videos, client=chat_client)

    original_fetch = rag_module.fetch_transcript
    transcripts_by_id = {"vidA": transcript_a, "vidB": transcript_b}
    rag_module.fetch_transcript = lambda vid, **kw: transcripts_by_id.get(vid)

    try:
        answer = engine.chat("What minimizes loss?", videos[0])
        _check(bool(answer), "chat returns a non-empty answer")

        summary = engine.summarize(videos[0], depth="eli5")
        _check(bool(summary), "summary returns non-empty text")

        quiz_script = [
            {
                "questions": [
                    {
                        "concept_id": concepts_a[0].id,
                        "question": "Why does gradient descent work?",
                        "answer": "It follows the negative gradient of the loss.",
                    }
                ]
            }
        ]
        engine._client = FakeClient(quiz_script)
        quiz = engine.generate_quiz(videos[0])
        _check(len(quiz.questions) >= 1, "quiz generates at least one question")
    finally:
        rag_module.fetch_transcript = original_fetch

    shutil.rmtree(data_dir, ignore_errors=True)
    return not _FAILURES


if __name__ == "__main__":
    ok = run()
    print("\nDRY RUN " + ("PASSED" if ok else f"FAILED ({len(_FAILURES)} check(s))"))
    sys.exit(0 if ok else 1)
