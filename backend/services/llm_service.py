import os
from dotenv import load_dotenv
from openai import OpenAI

from backend.config.config import EMBED_MODEL, CHAT_MODEL

load_dotenv()
_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(text: str) -> list:
    res = _client.embeddings.create(model=EMBED_MODEL, input=text)
    return res.data[0].embedding

def chat(messages, temperature: float = 0.2) -> str:
    res = _client.chat.completions.create(
        model=CHAT_MODEL,
        messages=messages,
        temperature=temperature
    )
    return res.choices[0].message.content
