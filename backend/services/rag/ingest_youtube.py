# services/rag/ingest_youtube.py
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
from services.rag.vectorstore import get_vectorstore

def split_text_into_chunks(text: str, max_words: int = 50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), max_words):
        chunk = " ".join(words[i:i+max_words])
        chunks.append(chunk)
    return chunks

def ingest_youtube(url: str):
    # Extract video ID
    video_id = url.split("v=")[-1].split("&")[0]

    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(video_id)  # returns FetchedTranscript object

        # Combine segments into one string
        text = " ".join([segment.text for segment in fetched_transcript]).strip()
        if not text:
            return []

        # Split text into chunks
        chunks = split_text_into_chunks(text)

        # Add to vectorstore
        vs = get_vectorstore()
        vs.add_documents(chunks)

        return chunks
    except (TranscriptsDisabled, NoTranscriptFound):
        return []
