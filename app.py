"""Streamlit UI: chronological playlist navigation, per-video completeness checklist,
chat scoped to "this video and everything before it", adjustable explanation depth, and
checklist-driven quizzes (so quiz coverage is auditable against the checklist).
"""
import time

import streamlit as st

from src import config
from src.concept_extractor import extract_concepts, load_manifest, save_manifest
from src.fetch_playlist import fetch_playlist_videos
from src.fetch_transcripts import fetch_transcript
from src.rag_engine import RagEngine, Video
from src.vectorstore import VectorStore

st.set_page_config(page_title="Playlist RAG Study Tool", layout="wide")


@st.cache_resource
def get_store() -> VectorStore:
    return VectorStore()


def ingest_playlist(playlist_url: str) -> None:
    store = get_store()
    videos = fetch_playlist_videos(playlist_url)

    progress = st.progress(0.0, text="Starting ingestion...")
    total_input_tokens = 0
    total_output_tokens = 0
    skipped = []
    started = time.monotonic()

    for i, v in enumerate(videos):
        progress.progress(i / len(videos), text=f"{v.title}")
        if store.has_video(v.video_id):
            continue

        transcript = fetch_transcript(v.video_id)
        if transcript is None:
            skipped.append(v.title)
            continue

        concepts, stats = extract_concepts(transcript)
        save_manifest(v.video_id, concepts)
        store.add_concepts(v.video_id, v.playlist_index, concepts)
        total_input_tokens += stats.input_tokens
        total_output_tokens += stats.output_tokens

    progress.progress(1.0, text="Done.")
    elapsed = time.monotonic() - started
    cost = config.estimate_cost_usd(config.CLAUDE_MODEL, total_input_tokens, total_output_tokens)

    st.session_state["videos"] = videos
    st.success(
        f"Ingested {len(videos) - len(skipped)}/{len(videos)} videos in {elapsed:.1f}s - "
        f"est. cost ${cost:.4f} ({total_input_tokens} in / {total_output_tokens} out tokens)."
    )
    if skipped:
        st.warning(f"No transcript available for {len(skipped)} video(s): {', '.join(skipped)}")


st.title("Playlist RAG Study Tool")

with st.sidebar:
    playlist_url = st.text_input("Playlist URL")
    if st.button("Ingest playlist") and playlist_url:
        ingest_playlist(playlist_url)

videos = st.session_state.get("videos", [])
if not videos:
    st.info("Paste a playlist URL in the sidebar and click Ingest to get started.")
    st.stop()

video_labels = [f"{i + 1}. {v.title}" for i, v in enumerate(videos)]
selected = st.sidebar.selectbox("Video", options=range(len(videos)), format_func=lambda i: video_labels[i])
current = videos[selected]
current_video = Video(video_id=current.video_id, title=current.title, playlist_index=current.playlist_index)

engine = RagEngine(get_store(), [Video(v.video_id, v.title, v.playlist_index) for v in videos])

checklist_tab, chat_tab, quiz_tab = st.tabs(["Checklist", "Chat", "Quiz"])

with checklist_tab:
    depth = st.radio("Depth", ["eli5", "standard", "deep"], horizontal=True, index=1)
    concepts = load_manifest(current.video_id)
    if not concepts:
        st.info("No checklist yet for this video - ingest the playlist first.")
    else:
        for c in concepts:
            st.checkbox(f"[{c.tag}] {c.point} ({int(c.start_seconds)}s)", key=f"{current.video_id}-{c.id}")
        if st.button("Summarize at this depth"):
            st.write(engine.summarize(current_video, depth))

with chat_tab:
    query = st.text_input("Ask about this video and everything before it")
    if query:
        st.write(engine.chat(query, current_video))

with quiz_tab:
    if st.button("Generate quiz"):
        quiz = engine.generate_quiz(current_video)
        if not quiz.questions:
            st.info("No core/supporting checklist items to quiz on yet.")
        for q in quiz.questions:
            with st.expander(q.question):
                st.write(q.answer)
