from sentence_transformers import SentenceTransformer
from beir import util

def load_embedding_model(model_name):
    model = SentenceTransformer(model_name)
    return model

def retrieve_top_k(model, query, corpus, k=10):
    query_embedding = model.encode(query, convert_to_tensor=True)
    corpus_embeddings = model.encode([corpus[doc_id]["text"] for doc_id in corpus.keys()], convert_to_tensor=True)

    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=k)[0]
    top_k_passages = [corpus[list(corpus.keys())[hit['corpus_id']]]["text"] for hit in hits]
    return top_k_passages
