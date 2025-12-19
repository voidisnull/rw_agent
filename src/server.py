import os

import uvicorn
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, Request, WebSocket
from fastapi.responses import HTMLResponse
from loguru import logger
from pipecat.frames.frames import StartFrame, EndFrame
from src.bot import LLM, STT
from src.memory import init_db
from src.server_utils import (
    DialoutResponse,
    dialout_request_from_request,
    generate_twiml,
    make_twilio_call,
    parse_twiml_request,
)

_ = load_dotenv(override=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Services")

    start_frame = StartFrame(audio_in_sample_rate=8000, audio_out_sample_rate=8000)

    try:
        logger.info("Initiazlizing db...")
        await init_db()

        logger.info("Starting LLM service...")
        await LLM.start(start_frame)

        logger.info("Starting STT service...")
        await STT.start(start_frame)

        logger.info("All services started successfully")

    except Exception:
        logger.error("Error while starting the services!!")

    yield

    logger.info("Server shutting down, cleaning up services...")
    end_frame = EndFrame()

    try:
        await LLM.stop(end_frame)
        await STT.stop(end_frame)
        logger.info("Services cleaned up successfully")
    except Exception:
        logger.error("Error while shutting down services!!")


app = FastAPI(lifespan=lifespan)


@app.post("/dialout", response_model=DialoutResponse)
async def handle_dialout_request(request: Request) -> DialoutResponse:
    """Handle outbound call request and initiate call via Twilio.

    Args:
        request (Request): FastAPI request containing JSON with 'to_number' and 'from_number'.

    Returns:
        DialoutResponse: Response containing call_sid, status, and to_number.

    Raises:
        HTTPException: If request data is invalid or missing required fields.
    """
    logger.info("Received outbound call request")

    dialout_request = await dialout_request_from_request(request)

    call_result = await make_twilio_call(dialout_request)

    return DialoutResponse(
        call_sid=call_result.call_sid,
        status="call_initiated",
        to_number=call_result.to_number,
    )


@app.post("/twiml")
async def get_twiml(request: Request) -> HTMLResponse:
    """Return TwiML instructions for connecting call to WebSocket.

    This endpoint is called by Twilio when a call is initiated. It returns TwiML
    that instructs Twilio to connect the call to our WebSocket endpoint with
    stream parameters containing call metadata.

    Args:
        request (Request): FastAPI request containing Twilio form data with 'To' and 'From'.

    Returns:
        HTMLResponse: TwiML XML response with Stream connection instructions.
    """
    logger.info("Serving TwiML for outbound call")

    twiml_request = await parse_twiml_request(request)

    twiml_content = generate_twiml(twiml_request)

    return HTMLResponse(content=twiml_content, media_type="application/xml")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Handle WebSocket connection from Twilio Media Streams.

    This endpoint receives the WebSocket connection from Twilio's Media Streams
    and runs the bot to handle the voice conversation. Stream parameters passed
    from TwiML are available to the bot for customization.

    Args:
        websocket (WebSocket): FastAPI WebSocket connection from Twilio.
    """
    from src.bot import bot
    from pipecat.runner.types import WebSocketRunnerArguments

    await websocket.accept()
    logger.info("WebSocket connection accepted for outbound call")

    try:
        runner_args = WebSocketRunnerArguments(websocket=websocket)
        await bot(runner_args)
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint: {e}")
        await websocket.close()


if __name__ == "__main__":
    # Run the server
    port = int(os.getenv("PORT", "7860"))
    logger.info(f"Starting Twilio outbound chatbot server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
