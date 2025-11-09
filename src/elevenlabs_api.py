"""
Riverwood Voice Agent:
Groq Whisper for STT  +  Groq Llama for Hinglish normalization  +  ElevenLabs Flash for TTS
"""

import os
import logging
from collections.abc import AsyncIterator
from dotenv import load_dotenv
from elevenlabs.client import AsyncElevenLabs
from groq import Groq

# ---------------- CONFIG ----------------
WHISPER_MODEL_ID = "whisper-large-v3-turbo"
HINGLISH_MODEL_ID = "llama-3.1-8b-instant"
TTS_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_VOICE_ID = "cgSgspJ2msm6clMCkdW9"

# ----------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_ = load_dotenv()

# ---------------- INIT ------------------
try:
    eleven_client = AsyncElevenLabs()
    if os.getenv("ELEVENLABS_API_KEY") is None:
        logger.warning("ELEVENLABS_API_KEY not set. ElevenLabs calls will fail.")
except Exception as err:
    logger.critical(f"Failed to initialize ElevenLabs client: {err}")
    eleven_client = None

try:
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    logger.info("Groq client initialized.")
except Exception as err:
    logger.critical(f"Failed to initialize Groq client: {err}")
    groq_client = None


# ---------------- STT -------------------
async def convert_audio_to_text(audio_file: bytes) -> str | None:
    """
    Convert audio bytes to text using Groq Whisper STT.
    If text contains Devanagari, rephrase it into Hinglish using Groq Llama.
    """
    if not groq_client:
        logger.error("Groq client not initialized. STT skipped.")
        return None

    temp_path = "temp_audio.mp3"
    try:
        with open(temp_path, "wb") as f:
            f.write(audio_file)

        # Whisper transcription
        with open(temp_path, "rb") as f:
            transcription = groq_client.audio.transcriptions.create(
                model=WHISPER_MODEL_ID,
                file=f,
                response_format="text",
                language="hi",
            )

        raw_text = transcription.strip()

        if len(raw_text.split()) == 1:
            return None

        logger.info(f"Raw Whisper output: '{raw_text}'")

        # Detect Hindi (Devanagari Unicode range)
        if any("\u0900" <= ch <= "\u097f" for ch in raw_text):
            logger.info("Detected Devanagari text → running Hinglish normalization...")
            try:
                prompt = (
                    "You are a precise text converter that rewrites Hindi or English sentences "
                    "into natural Hinglish (Roman Hindi). "
                    "Use Roman script only — no Devanagari characters. "
                    "Preserve the original meaning and flow, but write it the way people actually speak "
                    "in North Indian daily conversations (a mix of Hindi and English words). "
                    "Keep tone casual yet clear — like someone politely talking on the phone. "
                    "Do not over-translate English terms that are commonly used in Hindi speech "
                    "(e.g., keep 'painting', 'site visit', 'meeting', etc.). "
                    "Avoid emojis, punctuation overload, or added commentary. "
                    "Keep spelling simple and readable for an Indian audience.\n\n"
                    f"Input: {raw_text}\n\n"
                    "Output (in natural Hinglish):"
                )

                response = groq_client.chat.completions.create(
                    model=HINGLISH_MODEL_ID,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                    max_tokens=256,
                )
                normalized = response.choices[0].message.content.strip()
                logger.info(f"Hinglish-normalized: '{normalized}'")
                return normalized.lower()
            except Exception as err:
                logger.error(f"Hinglish normalization failed: {err}")
                return raw_text.lower()
        else:
            return raw_text.lower()

    except Exception as err:
        logger.error(f"Whisper STT exception: {err}")
        return None
    finally:
        try:
            os.remove(temp_path)
        except FileNotFoundError:
            pass


# ---------------- TTS -------------------
async def convert_text_to_audio_stream(
    text_message: str,
) -> AsyncIterator[bytes] | None:
    """
    Convert text to speech stream using ElevenLabs Flash.
    """
    if not eleven_client:
        logger.error("ElevenLabs client not initialized. TTS skipped.")
        return None

    logger.info(f"TTS synth ({TTS_MODEL_ID}): '{text_message}'")
    try:
        audio_stream = eleven_client.text_to_speech.convert(
            voice_id=DEFAULT_VOICE_ID,
            text=text_message,
            model_id=TTS_MODEL_ID,
            output_format="mp3_44100_128",
            optimize_streaming_latency=4,
        )
        logger.info("TTS success.")
        return audio_stream
    except Exception as err:
        logger.error(f"TTS error: {err}")
        return None
