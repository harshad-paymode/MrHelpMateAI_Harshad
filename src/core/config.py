"""
config.py
All constants and settings (dataclasses + simple constants).
"""
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv
import os


load_dotenv()

#Set up Azure OpenAI for Evaluation


@dataclass
class Models:
    MODEL_GENERATOR: str = os.getenv("MODEL_GENERATOR", "mistral-small-latest")
    MODEL_BACKUP_1: str = os.getenv("MODEL_BACKUP_1", "")
    MODEL_BACKUP_2: str = os.getenv("MODEL_BACKUP_2", "")
    MODEL_MODERATION: str = os.getenv("MODEL_MODERATION", "open-mistral-nemo")
    MODEL_API_KEY: str = os.getenv("MISTRAL_API")
    MODEL_RERANKER:str = os.getenv("MODEL_RERANKER")
    MODEL_EMBED:str = os.getenv("MODEL_EMBED")


@dataclass
class Paths:
    CHROMA_PERSISTENT: str = os.getenv("CHROMA_PERSISTENT", "./chroma_persist")
    POLICY_PDF_PATH: str = os.getenv("POLICY_PDF_PATH", "")
    POLICY_PDF_PATH_OUTPUT: str = os.getenv("POLICY_PDF_PATH_OUTPUT", "")
    OUTPUT_CHUNK_PATH: str = os.getenv("OUTPUT_CHUNK_PATH", "chunks.json")

@dataclass
class RetrievalConfig:
    footer_fraction: float = float(os.getenv("FOOTER_FRACTION", 0.12))
    min_chunk_words: int = int(os.getenv("MIN_CHUNK_WORDS", 20))
    score_threshold: float = float(os.getenv("SCORE_THRESHOLD", 0.55))
    retriever_k: int = int(os.getenv("RETRIEVER_K", 10))
    chroma_collection_name: str = os.getenv("CHROMA_COLLECTION_NAME", "collection_insurance")

@dataclass
class small_talk:
    # Static constants used across modules (kept here per instructions)
    SMALL_TALK_PATTERNS = [
        r"how (are|r) (you|u)\b",
        r"how's it going\b",
        r"how are you doing\b",
        r"what('?s| is) up\b",
        r"what('?s| is) going on\b",
        r"how is life\b",
        r"who are you\b",
        r"what can you do\b",
        r"how can you help\b",
        r"what are you\b",
        r"are you (a )?bot\b",
        r"are you human\b",
        r"tell me a joke\b",
        r"say something funny\b",
        r"make me laugh\b",
    ]

# instantiate config objects
MODELS = Models()
PATHS = Paths()
RETRIEVAL = RetrievalConfig()
SMALL_TALK_ROUTER = small_talk()