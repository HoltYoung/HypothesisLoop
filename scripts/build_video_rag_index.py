# This script fetches the transcript of a YouTube video and
# saves it as a markdown file in the RAG knowledge folder.

# Note: You may need to install the youtube_transcript_api package:
# pip install youtube-transcript-api


from pathlib import Path
from youtube_transcript_api import YouTubeTranscriptApi

video_id = "dQw4w9WgXcQ"
out_path = Path("knowledge/video_notes") / f"{video_id}.md"
out_path.parent.mkdir(parents=True, exist_ok=True)

transcript = YouTubeTranscriptApi().fetch(video_id, languages=["en"])

lines = ["# YouTube Transcript", f"\n## Video ID\n{video_id}\n", "## Segments"]
for seg in transcript:
    start = round(seg.start, 1)
    lines.append(f"\n### {start} seconds\n{seg.text}")

out_path.write_text("\n".join(lines), encoding="utf-8")
print(f"Saved to {out_path}")
