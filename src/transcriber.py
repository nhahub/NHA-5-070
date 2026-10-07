import os

from dotenv import load_dotenv
from groq import Groq


load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

client = Groq(api_key=api_key)


def get_value(item, key, default=None):
    """Read a value from either an object or a dictionary."""

    if isinstance(item, dict):
        return item.get(key, default)

    return getattr(item, key, default)


def transcribe(audio_path: str) -> dict:
    """
    Transcribe an audio file using Groq Whisper.

    Returns:
        {
            "text": str,
            "language": str | None,
            "quality": float
        }
    """

    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )

    text = transcription.text.strip()

    language = get_value(
        transcription,
        "language",
        None,
    )

    segments = get_value(
        transcription,
        "segments",
        None,
    )

    if not segments:
        quality = 0.0

    else:
        scores = []

        for segment in segments:

            avg_logprob = get_value(
                segment,
                "avg_logprob",
                None,
            )

            no_speech_prob = get_value(
                segment,
                "no_speech_prob",
                None,
            )

            if avg_logprob is None:
                continue

            confidence = min(
                1.0,
                max(
                    0.0,
                    avg_logprob + 1.0,
                ),
            )

            if no_speech_prob is not None:
                confidence *= (
                    1.0 - no_speech_prob
                )

            scores.append(confidence)

        quality = (
            sum(scores) / len(scores)
            if scores
            else 0.0
        )

    return {
        "text": text,
        "language": language,
        "quality": round(quality, 4),
    }