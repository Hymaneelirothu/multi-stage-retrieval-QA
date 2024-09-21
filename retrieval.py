from sentence_transformers import SentenceTransformer, util

def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

def retrieve_top_k(model, query, corpus, k=10):
    if not corpus or len(corpus) == 0:
        raise ValueError("Corpus is empty, cannot retrieve passages.")

    # Get embeddings for the query and corpus
    query_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = model.encode([corpus[doc_id]["text"] for doc_id in corpus])

    # Perform semantic search
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=k)
    top_k_passages = [(corpus[list(corpus.keys())[hit['corpus_id']]]["text"]) for hit in hits[0]]
    
    return top_k_passages
