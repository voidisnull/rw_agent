import os
import aiosqlite
from dotenv import load_dotenv
from groq import Groq
from src.prompt import MEMORY_SUMMARY_PROMPT

_ = load_dotenv(override=True)

DB_PATH = "memory.db"

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        _ = await db.execute("""
            CREATE TABLE IF NOT EXISTS call_memory (
                phone TEXT PRIMARY KEY,
                last_summary TEXT
            )
        """)
        await db.commit()


async def get_summary(phone: str):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT last_summary FROM call_memory WHERE phone = ?", (phone,)
        )
        row = await cur.fetchone()
        return row[0] if row else None


async def save_summary(phone: str, summary: str):
    async with aiosqlite.connect(DB_PATH) as db:
        _ = await db.execute(
            "INSERT OR REPLACE INTO call_memory (phone, last_summary) VALUES (?, ?)",
            (phone, summary),
        )
        await db.commit()


async def summarize_conversation(conversation_messages, previous_summary):
    if previous_summary:
        user_msg = (
            f"PREVIOUS MEMORY:\n{previous_summary}\n\n"
            f"NEW CONVERSATION:\n{conversation_messages}\n\n"
            "Generate the updated memory note:"
        )
    else:
        user_msg = (
            f"NEW CONVERSATION:\n{conversation_messages}\n\nGenerate the memory note:"
        )

    completion = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[
            {"role": "system", "content": MEMORY_SUMMARY_PROMPT},
            {"role": "user", "content": user_msg},
        ],
    )

    return completion.choices[0].message.content


async def finalize_call(phone: str, conversation_messages):
    previous = await get_summary(phone)
    new_summary = await summarize_conversation(conversation_messages, previous)
    await save_summary(phone, new_summary)
    return new_summary
