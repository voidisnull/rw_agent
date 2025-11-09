"""
Conversational AI Agent using Groq Llama and ChromaDB memory.
Now with persistent session memory.
"""

import os
import logging
import chromadb
from collections import defaultdict
from dotenv import load_dotenv
from openai import OpenAI

# --- Constants ---
PRIMARY_MODEL_ID = "llama-3.3-70b-versatile"
INSIGHT_MODEL_ID = "llama-3.3-70b-versatile"
CHROMA_COLLECTION_NAME = "conversation_memory"

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
_ = load_dotenv()

# --- Persona Prompt ---
AGENT_SYSTEM_PROMPT = (
    "You are Miss Riverwood — the warm, modern, and professional voice representative for Riverwood Projects LLP. "
    "You live inside the world of Riverwood Estate, a residential township in Kharkhauda, Haryana. "
    "It’s a calm morning most days, construction work is steady — painting, road laying, plumbing, and landscaping are in progress. "
    "You’ve been talking with clients and engineers for months, helping them with updates, visits, and project details. "
    "You speak naturally and confidently, like someone who knows the work well.\n\n"
    "Your personality: friendly, clear, and genuinely helpful — the way Indians are when they want to make someone feel comfortable. "
    "Be warm and conversational, but not overly formal. Use Hinglish (Roman Hindi) — mix Hindi and English words smoothly, "
    "just like people in Delhi NCR naturally speak. Avoid overusing words like 'ji' or 'sir' repeatedly; use them only once when appropriate. "
    "Focus on sounding confident and relaxed rather than overly polite. "
    "Your voice should feel like talking to a capable, cheerful project coordinator.\n\n"
    "Scene context: Riverwood Estate is around 80% complete. Painting, plumbing, and roadwork are active. "
    "Customers often ask about plot updates, site visits, or schedules. "
    "If something isn’t clear, make a reasonable assumption that fits ongoing construction and continue naturally.\n\n"
    "Your communication style:\n"
    "• Reply in short Hinglish (Roman Hindi) sentences — 20 to 40 words max.\n"
    "• Sound modern, polite, and effortless — not scripted or overly deferential.\n"
    "• Avoid repetition of filler words like 'ji', 'haan ji', 'bilkul ji'. Instead, just use natural connectors like 'haan', 'theek hai', 'okay', etc.\n"
    "• Keep the tone human and balanced — professional, but friendly.\n"
    "• Never write narration or stage directions.\n"
    "• If you don’t know an exact detail, assume something reasonable or redirect politely.\n\n"
    "Example style:\n"
    "User: Kal ka painting ka kaam aaj shuru hua kya?\n"
    "Miss Riverwood: Haan, painting team ne aaj subah primer start kar diya. Kal tak first coat complete ho jayega. Kaam smooth chal raha hai.\n\n"
    "Remember: you are not answering tickets — you are continuing a friendly, professional conversation about real progress at Riverwood Estate."
)


# --- Setup Groq + Chroma ---
try:
    client = OpenAI(
        api_key=os.getenv("GROQ_API_KEY"), base_url="https://api.groq.com/openai/v1"
    )
    logger.info(f"Groq client initialized with model '{PRIMARY_MODEL_ID}'.")

    chroma_client = chromadb.PersistentClient(path="./chroma_data")
    memory_collection = chroma_client.get_or_create_collection(CHROMA_COLLECTION_NAME)
    logger.info(f"ChromaDB collection '{CHROMA_COLLECTION_NAME}' ready.")

except Exception as setup_error:
    logger.critical(f"Agent setup failed: {setup_error}")
    client = None
    chroma_client = None
    memory_collection = None

# --- In-memory persistent sessions ---
active_sessions = defaultdict(list)


def get_or_create_session(session_id: str):
    """Fetch or create a message history for this session."""
    history = active_sessions[session_id]

    # Keep only last 8 messages + system prompt (4 turns)
    if len(history) > 9:
        active_sessions[session_id] = [history[0]] + history[-8:]

    return active_sessions[session_id]


def clear_session(session_id: str):
    """Remove session history."""
    if session_id in active_sessions:
        del active_sessions[session_id]
        logger.info(f"Session {session_id} cleared from memory.")


# --- Persistent conversation generation ---
async def generate_agent_reply(session_id: str, user_text: str) -> str:
    """Generate a reply while preserving chat context for a session."""
    if not user_text or not client:
        logger.warning("Missing input or uninitialized components.")
        return "Sorry, I’m having trouble processing that."

    if len(user_text.strip()) < 3:
        logger.warning("Empty or too-short user input, skipping generation.")
        return ""

    try:
        history = get_or_create_session(session_id)

        # If new session, seed with system persona
        if not history:
            history.append({"role": "system", "content": AGENT_SYSTEM_PROMPT})

            # Retrieve relevant memory from Chroma
            results = memory_collection.query(query_texts=[user_text], n_results=3)

            memory_context = ""
            if results and results.get("documents"):
                docs = [d for d in results["documents"][0] if d]
                if docs:
                    memory_context = " ".join(docs)

                # Merge memory into the system prompt instead of a separate message
                enriched_prompt = (
                    f"{AGENT_SYSTEM_PROMPT}\n\n"
                    "Before replying, recall what you already know from earlier conversations:\n"
                    f"{memory_context}\n\n"
                    "Think of this as your own memory — details you personally remember about the customer's last visit or queries. "
                    "Use it naturally in your reply, as if you remember it from experience. "
                    "Keep the flow warm, human, and consistent with your Riverwood role."
                )

                # Replace old system message with enriched version
                history[0] = {"role": "system", "content": enriched_prompt}

        # Append user message
        history.append({"role": "user", "content": user_text})

        # Single LLM call with session context
        response = client.chat.completions.create(
            model=PRIMARY_MODEL_ID,
            messages=history,
            temperature=0.8,
            max_tokens=150,
        )

        reply = response.choices[0].message.content.strip()
        history.append({"role": "assistant", "content": reply})
        logger.info(f"Session {session_id} reply: '{reply}'")

        return reply

    except Exception as err:
        logger.error(f"Persistent conversation error: {err}")
        return "[pauses] Sorry, something went wrong while responding."


# --- Summarize & store session memory ---
async def summarize_session(session_id: str) -> str | None:
    """Summarize the entire session and store as an insight in Chroma."""
    if not client or not memory_collection:
        logger.warning("Client or Chroma not ready. Skipping summary.")
        return None

    session_history = active_sessions.get(session_id)
    if not session_history:
        logger.warning(f"No session found for {session_id}")
        return None

    # Collect recent conversation (skip system prompts)
    convo_text = "\n".join(
        f"{m['role']}: {m['content']}"
        for m in session_history
        if m["role"] in ("user", "assistant")
    )

    try:
        prompt = (
            "You are Miss Riverwood’s internal memory system for Riverwood Projects LLP. "
            "Write a short internal note (2–3 lines) in natural Hinglish (Roman Hindi) "
            "that helps her recall what the customer talked about in this chat. "
            "Include what the user asked or discussed (like painting, site visit, plot update, or payment), "
            "any kaam progress ya promises jo mention hue, "
            "and overall tone or expectation of the user. "
            "Keep it factual, crisp, and easy to read aloud later — "
            "no emojis, no fluff, and no translation into Hindi script.\n\n"
            f"{convo_text}\n\n"
            "Memory note:"
        )

        response = client.chat.completions.create(
            model=INSIGHT_MODEL_ID,
            messages=[{"role": "system", "content": prompt}],
            temperature=0.5,
            max_tokens=150,
        )

        insight_text = response.choices[0].message.content.strip()
        if not insight_text:
            logger.warning("Empty summary output.")
            return None

        memory_collection.add(
            ids=[f"mem_{hash(insight_text)}"],
            documents=[insight_text],
            metadatas=[{"session_id": session_id}],
        )
        logger.info(f"Stored session summary for {session_id}: '{insight_text}'")
        return insight_text

    except Exception as err:
        logger.error(f"Session summary failed: {err}")
        return None
