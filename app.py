import streamlit as st
from data_preparation import load_dataset
from retrieval import load_embedding_model, retrieve_top_k
from reranking import load_ranking_model, rerank

# Streamlit App
st.title("Multi-Stage Text Retrieval Pipeline for QA")

# Input question
query = st.text_input("Enter a question:")

# Select embedding model
embed_model_name = st.selectbox("Select Embedding Model for Candidate Retrieval", 
                                  ("sentence-transformers/all-MiniLM-L6-v2", "nvidia/nv-embedqa-e5-v5"))
embed_model = load_embedding_model(embed_model_name)

# Load dataset
corpus, queries, qrels = load_dataset(num_docs=100)

if query:
    st.write("Loading embedding model...")
    st.write("Loading dataset...")
    
    # Retrieve top-k passages
    st.write("Retrieving top-k passages...")
    try:
        top_k_passages = retrieve_top_k(embed_model, query, corpus, k=10)

        # Display retrieved passages
        st.write("Top-k passages before reranking:")
        for passage in top_k_passages:
            st.write(passage)

        # Select ranking model
        rank_model_name = st.selectbox("Select Ranking Model for Re-Ranking", 
                                        ("cross-encoder/ms-marco-MiniLM-L-12-v2", "nvidia/nv-rerankqa-mistral-4b-v3"))
        rank_model, rank_tokenizer = load_ranking_model(rank_model_name)

        # Rerank the passages
        st.write("Reranking passages...")
        reranked_passages = rerank(rank_model, rank_tokenizer, query, top_k_passages)

        # Display reranked passages
        st.write("Reranked passages:")
        for passage in reranked_passages:
            st.write(passage)

    except Exception as e:
        st.write(f"Error during retrieval: {e}")
