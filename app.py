import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from sklearn.metrics import ndcg_score

# Helper function to load the dataset
def download_and_extract_dataset():
    import urllib.request
    import zipfile
    import os

    dataset_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"
    dataset_zip_path = "nq.zip"
    data_path = "./datasets/nq"

    # Download the dataset if not already downloaded
    if not os.path.exists(dataset_zip_path):
        st.write("Downloading the dataset... This may take a few minutes.")
        urllib.request.urlretrieve(dataset_url, dataset_zip_path)
        st.write("Download complete!")

    # Unzip the dataset if not already unzipped
    if not os.path.exists(data_path):
        st.write("Unzipping the dataset...")
        with zipfile.ZipFile(dataset_zip_path, 'r') as zip_ref:
            zip_ref.extractall("./datasets")
        st.write("Dataset unzipped!")

    return data_path

# Function to load corpus, queries, and qrels
def load_dataset():
    from beir.datasets.data_loader import GenericDataLoader
    data_path = download_and_extract_dataset()
    corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
    return corpus, queries, qrels

# Stage 1: Candidate retrieval using Sentence Transformer
def candidate_retrieval(query, corpus, top_k=10):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    corpus_ids = list(corpus.keys())
    corpus_embeddings = model.encode([corpus[doc_id]['text'] for doc_id in corpus_ids], convert_to_tensor=True)

    query_embedding = model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=top_k)[0]
    
    retrieved_docs = [corpus_ids[hit['corpus_id']] for hit in hits]
    return retrieved_docs

# Stage 2: Reranking using cross-encoder
def rerank(retrieved_docs, query, corpus, top_k=5):
    tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")
    model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-12-v2")

    scores = []
    for doc_id in retrieved_docs:
        text = corpus[doc_id]['text']
        inputs = tokenizer(query, text, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        scores.append(outputs.logits.item())

    reranked_indices = np.argsort(scores)[::-1][:top_k]
    reranked_docs = [retrieved_docs[idx] for idx in reranked_indices]
    return reranked_docs, scores

# Function to evaluate using NDCG@10
def evaluate_ndcg(reranked_docs, qrels, query_id, k=10):
    true_relevance = [qrels.get((query_id, doc_id), 0) for doc_id in reranked_docs]
    ideal_relevance = sorted(true_relevance, reverse=True)
    
    # NDCG expects input as 2D arrays
    return ndcg_score([ideal_relevance], [true_relevance], k=k)

# Streamlit main function
def main():
    st.title("Multi-Stage Retrieval Pipeline with Evaluation")

    st.write("Loading the dataset...")
    corpus, queries, qrels = load_dataset()
    st.write(f"Corpus Size: {len(corpus)}")
    
    # User input for asking a question
    user_query = st.text_input("Ask a question:")
    
    if user_query:
        st.write(f"Your query: {user_query}")
        
        st.write("Running Candidate Retrieval...")
        retrieved_docs = candidate_retrieval(user_query, corpus, top_k=10)
        
        st.write("Running Reranking...")
        reranked_docs, rerank_scores = rerank(retrieved_docs, user_query, corpus, top_k=5)
        
        st.write("Top Reranked Documents:")
        for doc_id in reranked_docs:
            st.write(f"Document ID: {doc_id}")
            st.write(f"Document Text: {corpus[doc_id]['text'][:500]}...")  # Show the first 500 characters of the document
        
        # Evaluation if the user query exists in the qrels (ground truth relevance labels)
        query_id = list(queries.keys())[0]  # Dummy query ID for now
        if query_id in queries:
            ndcg_score_value = evaluate_ndcg(reranked_docs, qrels, query_id, k=10)
            st.write(f"NDCG@10 Score: {ndcg_score_value}")
        else:
            st.write("No ground truth available for this query.")
        
        st.write("Query executed successfully!")

if __name__ == "__main__":
    main()
