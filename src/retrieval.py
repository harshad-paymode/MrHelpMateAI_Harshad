from .core.models import get_reranker
import pandas as pd

def format_docs_with_metadata(results):
    results_dict = {}
    #Keeping only the relevant results to the query
    for doc in results:
        results_dict.setdefault('Documents', []).append(doc.page_content)
        results_dict.setdefault('ID', []).append(doc.id)
        results_dict.setdefault('Metadatas', []).append(doc.metadata)

    results_df = pd.DataFrame(results_dict)
    return results_df


#These are small neural networks designed for reranking task.

def rerank_documents(query, results_df):
    """Re-rank retrieved docs using CrossEncoder."""
    
    # Get query from context (will be passed via RunnableParallel)
    # This is a simplification; see full example below for proper context flow
    cross_inputs = [[query, response] for response in results_df['Documents']]
    cross_encoder = get_reranker()
    cross_rerank_scores = cross_encoder.predict(cross_inputs)
    
    # Sort by score descending
    results_df['Reranked_scores'] = cross_rerank_scores
    
    results_df = results_df.sort_values(by='Reranked_scores', ascending=False)
    top_3_results = results_df[:3]

    return top_3_results[['Documents','Metadatas']]


