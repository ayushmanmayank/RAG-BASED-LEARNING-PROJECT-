"""Pulls a YouTube playlist's video list in strict playlist order via yt-dlp."""
from dataclasses import dataclass
from typing import List, Optional

import yt_dlp


@dataclass
class PlaylistVideo:
    video_id: str
    title: str
    url: str
    playlist_index: int
    duration: Optional[float]


def fetch_playlist_videos(playlist_url: str) -> List[PlaylistVideo]:
    ydl_opts = {
        "extract_flat": "in_playlist",
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(playlist_url, download=False)

    entries = info.get("entries") or []
    videos: List[PlaylistVideo] = []
    for i, entry in enumerate(entries):
        if not entry:
            # Private/deleted videos surface as None (or empty) placeholders in the
            # playlist listing - skip them rather than crashing the whole ingestion.
            continue
        video_id = entry.get("id")
        if not video_id:
            continue
        videos.append(
            PlaylistVideo(
                video_id=video_id,
                title=entry.get("title") or video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                playlist_index=i,
                duration=entry.get("duration"),
            )
        )

    if not videos:
        raise ValueError(f"No videos found in playlist: {playlist_url}")

    return videos
