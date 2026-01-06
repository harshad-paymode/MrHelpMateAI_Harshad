import os

from sentence_transformers import CrossEncoder
from langchain_mistralai import MistralAIEmbeddings
from langchain_chroma import Chroma
from langchain_mistralai import ChatMistralAI

from .config import MODELS, PATHS
from .logging_config import logger

def get_retriever():
    embeddings = MistralAIEmbeddings(model=MODELS.MODEL_EMBED)
    insurance_collection = Chroma(
    collection_name="collection_insurance",
    embedding_function=embeddings,
    persist_directory=PATHS.CHROMA_PERSISTENT,  # Where to save data locally, remove if not necessary
    )
    chroma_retriever = insurance_collection.as_retriever(
    search_type = "similarity_score_threshold",
    search_kwargs = {"k":10, "score_threshold":0.55}  #Higher score -> hihger similarity
    )
    return chroma_retriever

def get_generator():
    llm_gen = ChatMistralAI(
        model= MODELS.MODEL_GENERATOR,
        temperature=0,
        max_retries=2
        ).with_fallbacks(
            [
                ChatMistralAI(model = MODELS.MODEL_BACKUP_1, temperature = 0, max_retries=2),
                ChatMistralAI(model = MODELS.MODEL_BACKUP_2, temperature = 0, max_retries=2)
            ]
        )
    return llm_gen

def get_reranker():
    # Load cross encoder once
    cross_encoder = CrossEncoder(MODELS.MODEL_RERANKER)
    return cross_encoder

def get_moderator():
    mod_llm = ChatMistralAI(model=MODELS.MODEL_MODERATION)
    return mod_llm
