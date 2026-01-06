from .core.config import SMALL_TALK_ROUTER
import re
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import AIMessage

def is_small_talk(q: str):
    q = q.lower().strip()
    return any(re.search(p, q) for p in SMALL_TALK_ROUTER.SMALL_TALK_PATTERNS)

def get_router_message():
    small_talk = RunnableLambda(
        lambda x: AIMessage(content="Hi! I can help with insurance policy questions :)")
    )
    return small_talk
