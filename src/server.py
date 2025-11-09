"""
FastAPI backend for Riverwood Voice Agent.
Handles persistent LLM sessions, TTS playback, and session summary storage.
OPTIMIZED: Pipeline endpoint + background summary for faster response times.
"""

import logging
import asyncio
from fastapi import FastAPI, Form, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

# --- Local imports ---
from src.agent_logic import generate_agent_reply, summarize_session, clear_session
from src.elevenlabs_api import convert_text_to_audio_stream

# --- Setup logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Initialize app ---
app = FastAPI(title="Riverwood Voice Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Track background tasks ---
background_tasks: dict[str, asyncio.Task] = {}


# --- NEW: PIPELINE ENDPOINT (FASTEST) ---
@app.post("/process-audio")
async def process_audio_pipeline(
    session_id: str = Form(...), audio: UploadFile = File(...)
):
    """
    Pipeline endpoint: STT → LLM → TTS in one call.
    Eliminates 2 HTTP round trips for ~300-500ms improvement.
    """
    try:
        # 1. STT - Convert audio to text
        audio_bytes = await audio.read()
        from src.elevenlabs_api import convert_audio_to_text

        text = await convert_audio_to_text(audio_bytes)

        # Check if we got valid text
        if not text or len(text.strip()) < 3:
            logger.warning("STT returned empty or too short text, skipping.")
            return JSONResponse({"error": "No valid speech detected"}, status_code=400)

        logger.info(f"Pipeline STT: '{text}'")

        # 2. LLM - Generate reply
        reply = await generate_agent_reply(session_id, text)

        if not reply:
            return JSONResponse(
                {"error": "Agent failed to generate reply"}, status_code=500
            )

        logger.info(f"Pipeline LLM: '{reply}'")

        # 3. TTS - Convert to audio stream
        audio_stream = await convert_text_to_audio_stream(reply)

        if not audio_stream:
            return JSONResponse({"error": "TTS failed"}, status_code=500)

        # Return audio stream with metadata in headers
        return StreamingResponse(
            audio_stream,
            media_type="audio/mpeg",
            headers={"X-Transcript": text, "X-Reply": reply},
        )

    except Exception as err:
        logger.error(f"Pipeline error: {err}")
        return JSONResponse({"error": str(err)}, status_code=500)


# --- LEGACY ENDPOINTS (kept for backward compatibility) ---


@app.post("/chat")
async def chat(session_id: str = Form(...), user_text: str = Form(...)):
    """Handle user input and return agent reply (keeps session context)."""
    try:
        reply = await generate_agent_reply(session_id, user_text)
        return JSONResponse({"reply": reply})
    except Exception as err:
        logger.error(f"Chat error: {err}")
        return JSONResponse(
            {"error": "Agent failed to generate reply."}, status_code=500
        )


@app.post("/tts")
async def tts(text: str = Form(...)):
    """Convert text to audio stream via ElevenLabs."""
    try:
        audio_stream = await convert_text_to_audio_stream(text)
        if not audio_stream:
            return JSONResponse({"error": "TTS failed."}, status_code=500)
        return StreamingResponse(audio_stream, media_type="audio/mpeg")
    except Exception as err:
        logger.error(f"TTS error: {err}")
        return JSONResponse({"error": "TTS internal error."}, status_code=500)


@app.post("/stt")
async def stt(audio: UploadFile = File(...)):
    """Transcribe uploaded audio using Groq Whisper STT."""
    try:
        audio_bytes = await audio.read()
        from src.elevenlabs_api import convert_audio_to_text

        text = await convert_audio_to_text(audio_bytes)
        if not text:
            return JSONResponse({"error": "STT returned no text."}, status_code=400)

        logger.info(f"STT transcription: '{text}'")
        return JSONResponse({"text": text})
    except Exception as err:
        logger.error(f"STT error: {err}")
        return JSONResponse(
            {"error": "Speech-to-text conversion failed."}, status_code=500
        )


# --- OPTIMIZED: Background session summary ---
@app.post("/end")
async def end_session(session_id: str = Form(...)):
    """
    End session immediately, summarize in background.
    User gets instant "Call Ended" response.
    """
    try:
        # Clear session immediately
        clear_session(session_id)

        # Start summary in background (fire and forget)
        task = asyncio.create_task(background_summarize(session_id))
        background_tasks[session_id] = task

        # Return immediately - don't wait for summary
        logger.info(f"Session {session_id} ended, summary running in background")
        return JSONResponse({"message": "Session ended"})

    except Exception as err:
        logger.error(f"Session end error: {err}")
        return JSONResponse({"error": "Failed to end session."}, status_code=500)


async def background_summarize(session_id: str):
    """Background task to summarize and store session."""
    try:
        logger.info(f"Starting background summary for {session_id}")
        await summarize_session(session_id)
        logger.info(f"Background summary completed for {session_id}")
    except Exception as err:
        logger.error(f"Background summary failed for {session_id}: {err}")
    finally:
        # Cleanup task reference
        if session_id in background_tasks:
            del background_tasks[session_id]


# --- Cleanup background tasks on shutdown ---
@app.on_event("shutdown")
async def cleanup_background_tasks():
    """Cancel all pending background tasks on shutdown."""
    logger.info("Cancelling background tasks...")
    for task in background_tasks.values():
        task.cancel()
    await asyncio.gather(*background_tasks.values(), return_exceptions=True)
    logger.info("Background tasks cleaned up")


# --- Web UI route ---
@app.get("/", response_class=FileResponse)
async def home():
    """Serve static web interface."""
    return FileResponse("src/static/index.html")
