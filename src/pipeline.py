from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch
from langchain_community.retrievers import BM25Retriever
from langchain_core.messages import AIMessage
from .retrieval import format_docs_with_metadata, rerank_documents, get_retrieved_docs
from .routing import is_small_talk

def chain_elements(prompt, llm_gen):

    fallback = RunnableLambda(
        lambda x: {
            "answer": AIMessage(content="Sorry! This information is unavailable in the knowledge base."),
            "retrieval_context": [],
            "metadatas": []
        }
    )

    normal = (
        RunnableLambda(lambda x: {
            "query": x["query"],
            "bm25": BM25Retriever.from_documents(x["retrieved"], k=len(x["retrieved"])).invoke(x["query"])
        })
        | RunnableLambda(lambda x: {
            "query": x["query"],
            "results_df": format_docs_with_metadata(x["bm25"])
        })
        | RunnableLambda(lambda x: {
            "query": x["query"],
            "top": rerank_documents(x["query"], x["results_df"])
        })
        | RunnableLambda(lambda x: {
            "query": x["query"],
            "context": x["top"]["Documents"],
            "metadatas": x["top"]["Metadatas"]
        })
        | RunnableLambda(lambda x: {
            "prompt": prompt.invoke(x),
            "context": x["context"],
            "metadatas": x["metadatas"]
        })
        | RunnableLambda(lambda x: {
            "answer": llm_gen.invoke(x["prompt"]),
            "context": x["context"],
            "metadatas": x["metadatas"]
        })
    )

    return fallback, normal


def execute_chain(query, prompt, llm_gen, moderator, block_chain, small_talk):

    fallback, normal = chain_elements(prompt, llm_gen)

    moderation_chain = block_chain | RunnableLambda(
        lambda x: {"flag": "moderation", "response": x}
    )

    small_talk_chain = small_talk | RunnableLambda(
        lambda x: {"flag": "small_talk", "response": x}
    )

    fallback_chain = fallback | RunnableLambda(
        lambda x: {"flag": "no_results", "response": x}
    )

    normal_chain = normal | RunnableLambda(
        lambda x: {"flag": "rag", "response": x}
    )

    chain = RunnableBranch(
        (lambda x: not moderator.invoke({"input": x["query"]}), moderation_chain),
        (lambda x: is_small_talk(x["query"]), small_talk_chain),
        (
            RunnableParallel({
                "query": lambda x: x["query"],
                "retrieved": lambda x: get_retrieved_docs(x["query"])
            })
            | RunnableBranch(
                (lambda x: not x["retrieved"], fallback_chain),
                normal_chain
            )
        )
    )

    return chain.invoke({"query": query})
