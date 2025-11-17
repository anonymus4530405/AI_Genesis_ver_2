import uuid
from services.rag.vectorstore import get_vectorstore
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled

vs = get_vectorstore()

def ingest_youtube(video_id: str, chunk_size: int = 800):
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    except TranscriptsDisabled:
        return {"status": "failed", "reason": "Transcripts disabled for this video"}
    except Exception as e:
        return {"status": "failed", "reason": str(e)}

    buffer = ""
    chunks_count = 0

    for line in transcript:
        buffer += " " + line["text"]
        if len(buffer) >= chunk_size:
            vs.add_documents(
                docs=[buffer.strip()],
                ids=[str(uuid.uuid4())],
                payloads=[{
                    "source_type": "youtube",
                    "video_id": video_id,
                    "timestamp": line["start"]
                }]
            )
            chunks_count += 1
            buffer = ""

    if buffer.strip():  # leftover text
        vs.add_documents(
            docs=[buffer.strip()],
            ids=[str(uuid.uuid4())],
            payloads=[{
                "source_type": "youtube",
                "video_id": video_id,
                "timestamp": transcript[-1]["start"]
            }]
        )
        chunks_count += 1

    return {"status": "success", "chunks": chunks_count}
