from .core.models import get_moderator
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage
from .prompts import get_moderation_temlate

def moderation_check():
    prompt = get_moderation_temlate()
    mod_llm =   get_moderator()
    
    moderator = (
        prompt
        | mod_llm
        | RunnableLambda(lambda x: "SAFE" in x.content)
    )

    block_chain = RunnableLambda(
        lambda x: AIMessage(content="I cannot answer harmful or inappropriate questions.")
    )
    return moderator,block_chain
