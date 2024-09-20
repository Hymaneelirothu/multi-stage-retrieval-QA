from sklearn.metrics import ndcg_score

def evaluate_ndcg(top_k_passages, qrels):
    relevance_scores = [1 if doc in qrels else 0 for doc, _ in top_k_passages]
    return ndcg_score([relevance_scores], [[1]*len(relevance_scores)], k=10)
