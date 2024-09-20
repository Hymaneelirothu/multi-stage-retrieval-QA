import streamlit as st
from data_preparation import load_dataset
from retrieval import load_embedding_model, retrieve_top_k
from reranking import load_ranking_model, rerank
from evaluation import evaluate_ndcg

# Set up the Streamlit interface
st.title("Multi-Stage Text Retrieval Pipeline for QA")

# Query Input
query = st.text_input("Enter a question:", "What is the capital of France?")

# Embedding model selection
embedding_model = st.selectbox(
    "Select Embedding Model for Candidate Retrieval",
    ["sentence-transformers/all-MiniLM-L6-v2", "nvidia/nv-embedqa-e5-v5"]
)

# Ranking model selection
ranking_model = st.selectbox(
    "Select Ranking Model for Re-Ranking",
    ["cross-encoder/ms-marco-MiniLM-L-12-v2", "nvidia/nv-rerankqa-mistral-4b-v3"]
)

# Run retrieval pipeline on button click
if st.button("Run Retrieval"):
    # Load dataset
    st.write("Loading dataset...")
    corpus, queries, qrels = load_dataset("nq")

    # Load selected embedding model
    st.write(f"Loading embedding model: {embedding_model}...")
    embed_model = load_embedding_model(embedding_model)

    # Retrieve top-k passages using embedding model
    st.write("Retrieving top-k passages...")
    top_k_passages = retrieve_top_k(embed_model, query, corpus, k=10)
    
    # Display retrieved passages
    st.write("Top-k passages before reranking:")
    for i, (passage, score) in enumerate(top_k_passages):
        st.write(f"{i+1}. Passage: {passage}, Score: {score:.4f}")

    # Load selected ranking model
    st.write(f"Loading ranking model: {ranking_model}...")
    rank_model, rank_tokenizer = load_ranking_model(ranking_model)

    # Rerank the retrieved passages
    st.write("Reranking passages...")
    ranked_passages = rerank(rank_model, rank_tokenizer, query, top_k_passages)
    
    # Display reranked passages
    st.write("Top-k passages after reranking:")
    for i, (passage, score) in enumerate(ranked_passages):
        st.write(f"{i+1}. Passage: {passage}, Score: {score:.4f}")

    # Evaluate using NDCG@10
    st.write("Evaluating NDCG@10...")
    query_id = list(queries.keys())[0]  # Assuming we are using the first query for evaluation
    ndcg_score = evaluate_ndcg(ranked_passages, qrels[query_id])
    st.write(f"NDCG@10: {ndcg_score:.4f}")

# Sidebar with instructions
st.sidebar.title("Instructions")
st.sidebar.write("""
1. Enter a question in the text input.
2. Select the embedding model for candidate retrieval.
3. Select the ranking model for reranking the retrieved passages.
4. Click 'Run Retrieval' to start the pipeline and display the results.
""")
