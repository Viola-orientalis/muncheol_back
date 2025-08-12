from pathlib import Path
import os

# Root of the repo (P_backend/)
ROOT_DIR   = Path(__file__).resolve().parents[2]

# Data dirs
DATA_DIR   = ROOT_DIR / "data"
RAW_DIR    = DATA_DIR / "raw"
CLEANED_DIR= DATA_DIR / "cleaned"
CHUNKS_DIR = DATA_DIR / "chunks"

# Vector store dir
VSTORE_DIR = ROOT_DIR / "vectorstore" / "dev" / "current"

# Embedding/LLM models
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
CHAT_MODEL  = os.getenv("CHAT_MODEL", "gpt-4o")

# Flask/CORS
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*")  # comma-separated in prod
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
