# Riverwood Voice Assistant (Hinglish Conversational AI)

**Miss Riverwood** is a bilingual (Hinglish) conversational AI assistant designed for **Riverwood Projects LLP**, a real-estate developer. It enables natural, voice-driven, emotionally intelligent conversations between customers and an AI representative that speaks in Hinglish (Roman Hindi).

This project demonstrates how modern **LLMs**, **speech APIs**, and **semantic memory** can create an **immersive, human-like conversational experience** without heavy infrastructure or GPU requirements.

- - -

## Features

*   **Natural Voice Interaction**
    Real-time **speech-to-text (STT)** and **text-to-speech (TTS)** pipeline using **Groq Whisper** and **ElevenLabs**.
*   **Memory-Driven Conversations**
    Retains session data and semantic insights via **ChromaDB**, so Miss Riverwood can recall past discussions and context.
*   **Persistent Dialogue Sessions**
    Each session maintains context across multiple exchanges for fluid, human-like continuity.
*   **Narrative-Driven Personality**
    Miss Riverwood isn’t just an assistant — she exists inside a believable narrative world, complete with personality, tone, and situational awareness.
*   **Modern Hinglish Voice**
    Fluent bilingual speech generation that feels truly local — English and Hindi mixed naturally, no formal translations.
*   **Composable Architecture**
    Modular, readable codebase split into components:
    *   `agent_logic.py`: Handles Groq LLM calls + context memory
    *   `elevenlabs_api.py`: Speech-to-text and text-to-speech
    *   `server.py`: FastAPI endpoints
    *   `static/index.html`: Tailwind + Alpine.js UI

## Architecture Overview
```
 ┌────────────────────────┐
 │       Frontend         │
 │  (Tailwind + AlpineJS) │
 │    Mic + Speaker UI    │
 └──────────┬─────────────┘
            │ Audio Stream
            ▼
 ┌────────────────────────┐
 │      FastAPI Server    │
 │   /transcribe  /speak  │
 └──────────┬─────────────┘
            │
     ┌──────┼──────────────────────────────┐
     ▼                                      ▼
┌─────────────┐                         ┌────────────────┐
│ STT Module  │                         │  Agent Logic   │
│(Groq Whisper)                         │  (Groq LLaMA)  │
│Speech → Text                          │  Context + LLM │
└─────────────┘                         └────────────────┘
            │                                     │
            ▼                                     ▼
     ┌────────────┐                         ┌─────────────────┐
     │  ChromaDB  │ ← Memory Notes          │  TTS (ElevenLabs) │
     └────────────┘                         └─────────────────┘
```
**Flow:**

1.  User speaks → STT → text.
2.  Text normalized to Hinglish.
3.  Context retrieved from ChromaDB (if available).
4.  Groq LLM generates Miss Riverwood’s reply.
5.  Insight model saves memory → TTS → speech output.

## Tech Stack

| Layer | Technology | Purpose |
| --- | --- | --- |
| Backend | FastAPI | API orchestration & routing |
| Frontend | TailwindCSS + Alpine.js | Lightweight reactive UI |
| LLM Engine | Groq (llama-3.1-70b-versatile) | Conversational reasoning & Hinglish responses |
| STT Engine | Groq Whisper | Hindi/English transcription |
| TTS Engine | ElevenLabs (eleven\_flash\_v2\_5) | Human-like Hinglish speech |
| Memory Store | ChromaDB (local) | Long-term vector memory |
| Runtime | Python 3.11 | Core execution environment |

## Alternative (Open Source) Stack

If you want a fully open-source version (no paid APIs):

| Function | OSS Alternative | Notes |
| --- | --- | --- |
| LLM | Ollama + LLaMA 3.1 8B | Local inference; slower but free |
| STT | Whisper.cpp | Local transcription engine |
| TTS | Coqui TTS or Svara TTS | Open-source neural speech |
| Memory | FAISS or Weaviate | Replace ChromaDB |
| Frontend | Svelte / Vue | Optional, modern UI frameworks |

This version can run entirely offline for research or academic use.

## Predicted Costs (Demo Scale)

| Component | Provider | Est. Cost | Notes |
| --- | --- | --- | --- |
| Groq LLaMA 70B | Groq Cloud | Free (100k tokens/day) | Within free tier |
| Groq Whisper STT | Groq Cloud | Free (1k mins/day) | No current cost |
| ElevenLabs TTS | ElevenLabs Starter | ~$5/month | ~10k–15k chars |
| ChromaDB | Local instance | Free | Persistent |
| Infra | Railway / Render | ~$5–10/month | Hosting container |

≈ $5–10 / month total for demo-scale deployments

Zero infrastructure lock-in — you can self-host everything later.

## Core Design Philosophy

Miss Riverwood is not a “bot.” She is a role-playing entity inside a simulated world.

The model is guided by a detailed narrative prompt that defines her environment, her tone, and her personality — warm, modern, and distinctly Indian.

This “story-driven agent” design allows the LLM to make reasonable assumptions, improvise naturally, and respond as if she genuinely knows the user and the project.

## Setup Instructions

1\. **Clone the repository**

```
git clone https://github.com/yourusername/riverwood-voice-assistant.git
cd riverwood-voice-assistant
```

2\. **Create a virtual environment**

```
uv venv
source .venv/bin/activate
```

3\. **Install dependencies**

```
uv pip install -r requirements.txt
```

4\. **Set up environment variables**

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
ELEVEN_API_KEY=your_elevenlabs_api_key
```

5\. **Run the backend server**

```
uvicorn src.server:app --reload
```

6\. **Access the frontend**

Visit: `http://localhost:8000`

You’ll see the web UI — tap the microphone circle to start a conversation.

## Conversation Flow

*   User taps mic → browser records audio.
*   Audio → Groq Whisper (STT).
*   Text → LLM (Groq LLaMA-3.1-70B).
*   Context fetched → personalized reply generated.
*   Memory updated → reply spoken via ElevenLabs TTS.
*   Loop continues until user ends session.

## Future Roadmap

*   **Realtime Voice Streaming**
    Full duplex mode with <300ms latency using WebSockets.
*   **Intent Classification Layer**
    Auto-detect queries like site update, payment, scheduling, documentation.
*   **Dynamic Knowledge Graph**
    Integration with real project CRM/ERP data for factual accuracy.
*   **LLM Fine-Tuning**
    Persona-adapted fine-tuning on Riverwood domain corpus.
*   **Multimodal Expansion**
    Add support for images, PDFs, and construction status charts.
*   **Offline Mode**
    Fully OSS pipeline (Ollama + Whisper.cpp + Coqui TTS) for local deployments.

## Architecture Summary

*   **Voice → Text → LLM → Voice**
    Modular async pipeline with memory persistence.
*   **Session Persistence**
    In-memory during call + ChromaDB for long-term memory.
*   **Hinglish Normalization Layer**
    Ensures linguistic consistency across STT and TTS.
*   **Warm Persona Design**
    Balanced tone: modern Indian, confident, caring.

## Project Impact

Riverwood Voice Assistant serves as a prototype for Indian bilingual customer support systems. It proves that LLMs can simulate empathy and continuity even without deep domain integration, through well-designed narrative prompts and contextual memory.

## Summary

Miss Riverwood is not just a conversational demo — it’s a vision of localized, emotionally intelligent AI for real estate and service industries in India.

By combining LLMs + voice + narrative design, it creates a believable human presence through technology.
