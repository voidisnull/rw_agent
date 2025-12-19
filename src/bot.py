import os
import sys

from dotenv import load_dotenv
from loguru import logger
from src.memory import finalize_call
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.pipeline.pipeline import Pipeline
from pipecat.transcriptions.language import Language
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContextAggregatorPair,
)
from pipecat.runner.types import RunnerArguments
from pipecat.runner.utils import parse_telephony_websocket
from pipecat.serializers.twilio import TwilioFrameSerializer

# from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.services.cartesia.tts import CartesiaTTSService, GenerationConfig
from pipecat.services.groq.stt import GroqSTTService
from pipecat.services.groq.llm import GroqLLMService
from pipecat.transports.base_transport import BaseTransport
from pipecat.transports.websocket.fastapi import (
    FastAPIWebsocketParams,
    FastAPIWebsocketTransport,
)


from src.prompt import SYSTEM_PROMPT

_ = load_dotenv(override=True)

logger.remove(0)
_ = logger.add(sys.stderr, level="DEBUG")

LLM = GroqLLMService(
    api_key=os.getenv("GROQ_API_KEY"),
    model="meta-llama/llama-4-scout-17b-16e-instruct",
)

STT = GroqSTTService(
    api_key=os.getenv("GROQ_API_KEY"),
    language=Language.HI,
    model="whisper-large-v3-turbo",
    temperature=0.15,
)

TTS = CartesiaTTSService(
    api_key=os.getenv("CARTESIA_API_KEY"),
    voice_id="95d51f79-c397-46f9-b49a-23763d3eaa2d",
    cartesia_version="2025-10-27",
    sample_rate=8000,
    params=CartesiaTTSService.InputParams(
        language=Language.HI,
        generation_config=GenerationConfig(emotion="enthusiastic"),
    ),
)

# TTS = ElevenLabsTTSService(
#     api_key=os.getenv("ELEVENLABS_API_KEY"),
#     voice_id="cgSgspJ2msm6clMCkdW9",
#     sample_rate=8000,
# )

VAD = SileroVADAnalyzer(
    sample_rate=8000, params=VADParams(stop_secs=0.35, start_secs=0.15)
)


async def run_bot(transport: BaseTransport, handle_sigint: bool):
    llm = LLM
    stt = STT
    tts = TTS

    # previous_summary = await get_summary(from_number)
    # memory_context = f"\n\nPREVIOUS CONTEXT:\n{previous_summary}" if previous_summary else ""

    # messages = [
    #     {
    #         "role": "system",
    #         "content": SYSTEM_PROMPT + memory_context,
    #     },
    # ]

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
    ]

    context = LLMContext(messages)
    context_aggregator = LLMContextAggregatorPair(context)

    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            context_aggregator.user(),
            llm,
            tts,
            transport.output(),
            context_aggregator.assistant(),
        ]
    )

    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            enable_metrics=True,
            enable_usage_metrics=True,
        ),
    )

    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # Kick off the outbound conversation, waiting for the user to speak first
        logger.info("Starting outbound call conversation")

    @transport.event_handler("on_client_disconnected")
    async def on_client_disconnected(transport, client):
        logger.info("Outbound call ended")

        phone = client.get("from_number")
        await finalize_call(phone, context.messages)

        await task.cancel()

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


async def bot(runner_args: RunnerArguments):
    """Main bot entry point compatible with Pipecat Cloud."""
    transport_type, call_data = await parse_telephony_websocket(runner_args.websocket)
    logger.info(f"Auto-detected transport: {transport_type}")

    # Access custom stream parameters passed from TwiML
    # Use the body data to personalize the conversation
    # by loading customer data based on the to_number or from_number
    body_data = call_data.get("body", {})
    to_number = body_data.get("to_number")
    from_number = body_data.get("from_number")

    logger.info(f"Call metadata - To: {to_number}, From: {from_number}")

    serializer = TwilioFrameSerializer(
        stream_sid=call_data["stream_id"],
        call_sid=call_data["call_id"],
        account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
        auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
    )

    transport = FastAPIWebsocketTransport(
        websocket=runner_args.websocket,
        params=FastAPIWebsocketParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            add_wav_header=False,
            vad_analyzer=VAD,
            serializer=serializer,
            audio_in_sample_rate=8000,
            audio_out_sample_rate=8000,
            audio_out_10ms_chunks=3,
        ),
    )

    handle_sigint = runner_args.handle_sigint

    await run_bot(transport, handle_sigint)
