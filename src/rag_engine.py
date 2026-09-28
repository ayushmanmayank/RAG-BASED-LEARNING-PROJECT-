"""Chat / summary / quiz generation via Claude, scoped to "this video and everything
before it" in the playlist (no spoilers from later videos).

Because the vectorstore never persists transcript text, every chat answer re-fetches the
actual words live from fetch_transcripts for whichever (video, timestamp) the vectorstore
points at, rather than reading cached text.
"""
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import anthropic
from pydantic import BaseModel

from src import config
from src.concept_extractor import load_manifest
from src.fetch_transcripts import fetch_transcript
from src.vectorstore import VectorStore

Depth = Literal["eli5", "standard", "deep"]

DEPTH_INSTRUCTIONS = {
    "eli5": "Explain like the listener is a total beginner. Avoid jargon, use simple analogies.",
    "standard": "Explain at the level of someone following the course normally.",
    "deep": "Give a thorough, technically precise explanation, including nuance and edge cases.",
}


@dataclass
class Video:
    video_id: str
    title: str
    playlist_index: int


class QuizQuestion(BaseModel):
    concept_id: int
    question: str
    answer: str


class Quiz(BaseModel):
    questions: List[QuizQuestion]


def _context_window_text(video_id: str, center_seconds: float, pad_seconds: float = 45.0) -> str:
    transcript = fetch_transcript(video_id)
    if transcript is None:
        return ""
    lines = [
        s.text
        for s in transcript.segments
        if center_seconds - pad_seconds <= s.start <= center_seconds + pad_seconds
    ]
    return " ".join(lines)


class RagEngine:
    def __init__(
        self,
        store: VectorStore,
        videos: List[Video],
        client: Optional[anthropic.Anthropic] = None,
    ):
        self._store = store
        self._videos: Dict[str, Video] = {v.video_id: v for v in videos}
        self._client = client or anthropic.Anthropic()

    def chat(self, query: str, current_video: Video) -> str:
        hits = self._store.search(query, max_playlist_index=current_video.playlist_index, n_results=6)

        context_chunks = []
        for hit in hits:
            words = _context_window_text(hit["video_id"], hit["start_seconds"])
            if not words:
                continue
            video = self._videos.get(hit["video_id"])
            label = video.title if video else hit["video_id"]
            context_chunks.append(f"[{label} @ {int(hit['start_seconds'])}s] {words}")

        system = (
            "You are a study assistant for a video course, answering strictly using the "
            "provided excerpts. Never reference or hint at material from videos later in "
            "the playlist than the current one. If the excerpts don't answer the question, "
            "say so instead of guessing."
        )
        context = "\n\n".join(context_chunks) or "(no relevant excerpts found)"
        prompt = f"Course excerpts so far:\n\n{context}\n\nQuestion: {query}"

        response = self._client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return next((b.text for b in response.content if b.type == "text"), "")

    def summarize(self, video: Video, depth: Depth = "standard") -> str:
        concepts = load_manifest(video.video_id)
        checklist = "\n".join(f"- ({c.tag}) {c.point}" for c in concepts)
        system = f"Summarize this video's checklist of points for a learner. {DEPTH_INSTRUCTIONS[depth]}"
        response = self._client.messages.create(
            model=config.CLAUDE_MODEL,
            max_tokens=2000,
            system=system,
            messages=[{"role": "user", "content": f"Video: {video.title}\n\nChecklist:\n{checklist}"}],
        )
        return next((b.text for b in response.content if b.type == "text"), "")

    def generate_quiz(self, video: Video, max_questions: int = 8) -> Quiz:
        concepts = [c for c in load_manifest(video.video_id) if c.tag in ("core", "supporting")][:max_questions]
        if not concepts:
            return Quiz(questions=[])

        checklist = "\n".join(f"{c.id}: ({c.tag}) {c.point}" for c in concepts)
        system = (
            "Write one quiz question per checklist item below, testing understanding of "
            "exactly that point. Use the item's id as concept_id so quiz coverage against "
            "the checklist is auditable."
        )
        response = self._client.messages.parse(
            model=config.CLAUDE_MODEL,
            max_tokens=4000,
            system=system,
            messages=[{"role": "user", "content": checklist}],
            output_format=Quiz,
        )
        return response.parsed_output
