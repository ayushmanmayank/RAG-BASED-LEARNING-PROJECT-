"""Fetches per-video transcripts from YouTube: manual captions preferred, auto-generated
fallback. Transcript text is never written to disk - callers re-fetch on demand, and this
module only keeps an in-memory cache for the lifetime of the process, so restarting the
process re-fetches from YouTube rather than reading stale local text.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import (
    NoTranscriptFound,
    TranscriptsDisabled,
    VideoUnavailable,
)

logger = logging.getLogger(__name__)

DEFAULT_LANGUAGES = ["en", "en-US", "en-GB"]


@dataclass
class TranscriptSegment:
    index: int
    start: float
    duration: float
    text: str


@dataclass
class Transcript:
    video_id: str
    language: str
    is_generated: bool
    segments: List[TranscriptSegment]

    @property
    def full_text(self) -> str:
        return " ".join(s.text for s in self.segments)


_cache: Dict[str, Optional[Transcript]] = {}


def _normalize(video_id: str, raw_snippets, language: str, is_generated: bool) -> Transcript:
    segments = []
    for i, s in enumerate(raw_snippets):
        text = (s.get("text") if isinstance(s, dict) else getattr(s, "text", "")) or ""
        text = text.strip()
        if not text:
            continue
        start = s.get("start") if isinstance(s, dict) else getattr(s, "start", 0.0)
        duration = s.get("duration") if isinstance(s, dict) else getattr(s, "duration", 0.0)
        segments.append(TranscriptSegment(index=i, start=float(start or 0.0), duration=float(duration or 0.0), text=text))
    return Transcript(video_id=video_id, language=language, is_generated=is_generated, segments=segments)


def _pick_transcript(transcript_list, preferred_languages: List[str]):
    manual = None
    generated = None
    for t in transcript_list:
        if not t.is_generated and manual is None:
            manual = t
        if t.is_generated and generated is None:
            generated = t

    for candidate in (manual, generated):
        if candidate is None:
            continue
        try:
            return candidate.find_transcript(preferred_languages)
        except Exception:
            return candidate

    for t in transcript_list:
        return t
    return None


def fetch_transcript(video_id: str, preferred_languages: Optional[List[str]] = None) -> Optional[Transcript]:
    """Returns None (never raises) when a video has no usable transcript at all."""
    if video_id in _cache:
        return _cache[video_id]

    preferred_languages = preferred_languages or DEFAULT_LANGUAGES
    result: Optional[Transcript] = None

    try:
        # youtube-transcript-api >= 0.6: instance-based API.
        api = YouTubeTranscriptApi()
        transcript_list = api.list(video_id)
        chosen = _pick_transcript(transcript_list, preferred_languages)
        if chosen is not None:
            fetched = chosen.fetch()
            raw = fetched.to_raw_data() if hasattr(fetched, "to_raw_data") else fetched
            result = _normalize(video_id, raw, chosen.language_code, chosen.is_generated)
    except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
        logger.warning("No transcript available for %s: %s", video_id, e)
        result = None
    except AttributeError:
        # Older youtube-transcript-api (<0.6): static-method API.
        try:
            raw = YouTubeTranscriptApi.get_transcript(video_id, languages=preferred_languages)
            result = _normalize(video_id, raw, preferred_languages[0], is_generated=True)
        except (TranscriptsDisabled, NoTranscriptFound, VideoUnavailable) as e:
            logger.warning("No transcript available for %s: %s", video_id, e)
            result = None
    except Exception as e:
        logger.warning("Transcript fetch failed for %s: %s", video_id, e)
        result = None

    _cache[video_id] = result
    return result
