from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch
from langchain_core.messages import AIMessage
from langchain_community.retrievers import BM25Retriever
from .retrieval import format_docs_with_metadata, rerank_documents
from .routing import is_small_talk

def chain_elements(prompt, llm_gen):
    fallback = RunnableLambda(
        lambda x: AIMessage(content="Sorry! This information is unavailable in the knowledge base.")
    )

    normal = (
        RunnableLambda(lambda x :{
                "query": x["query"],
                "bm-25-retrieved": BM25Retriever.from_documents(x['retrieved'], k=len(x['retrieved'])).invoke(x['query'])
            })
            | RunnableLambda(lambda x: {
                "query": x["query"],
                "results_df": format_docs_with_metadata(x["bm-25-retrieved"])
            })
            |
            RunnableLambda(lambda x: {
            "query": x["query"],
            "top_docs_and_context": rerank_documents(x["query"], x["results_df"])
            })
            | RunnableLambda (lambda x: {
                "query": lambda x: x["query"],
                "context" : x["top_docs_and_context"]["Documents"],
                "metadatas" : x["top_docs_and_context"]["Metadatas"]
            })
            | prompt 
            | llm_gen
    )
    return fallback, normal


def execute_chain(query, prompt,llm_gen, moderator,block_chain, small_talk, chroma_retriever):
    fallback,normal = chain_elements(prompt,llm_gen)

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
        # moderation first
        (lambda x: not moderator.invoke({"input": x["query"]}), moderation_chain),

        # small-talk
        (lambda x: is_small_talk(x["query"]), small_talk_chain),

        # RAG pipeline
        (
            RunnableParallel({
                "query": lambda x: x["query"],
                "retrieved": lambda x: chroma_retriever.invoke(x["query"]),
            })
            | RunnableBranch(
                # if results_df is empty → fallback
                (lambda x: not x["retrieved"], fallback_chain),
                # otherwise → continue normal path
                normal_chain,
            )
        ),
    )
    response = chain.invoke({"query": query})
    return response
    
