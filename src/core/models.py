import os

from sentence_transformers import CrossEncoder
from langchain_mistralai import ChatMistralAI
from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM

from .config import MODELS
from .logging_config import logger

#set up Azure OpenAI for evaluation

class AzureOpenAI(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model

    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return "gpt-40"


# Replace these with real values
def get_evaluator():
    custom_model = AzureChatOpenAI(
        openai_api_version=os.getenv("OPENAI_API_VERSION"),
        azure_deployment=os.getenv("OPENAI_MODEL"),
        azure_endpoint=os.getenv("AZURE_ENDPOINT"),
        openai_api_key=os.getenv("OPENAI_KEY"),
    )
    azure_openai = AzureOpenAI(model=custom_model)
    return azure_openai

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
    
    logger.info("LLM Generator is Created!")
    return llm_gen

def get_reranker():
    # Load cross encoder once
    cross_encoder = CrossEncoder(MODELS.MODEL_RERANKER)
    logger.info("Reranker Created!")
    return cross_encoder

def get_moderator():
    mod_llm = ChatMistralAI(model=MODELS.MODEL_MODERATION)
    logger.info("Moderator model is created!")
    return mod_llm
