import streamlit as st
from data_preparation import load_dataset
from retrieval import load_embedding_model, retrieve_top_k

# Title of the Streamlit App
st.title("Multi-Stage Text Retrieval Pipeline for QA")

# Input question
query = st.text_input("Enter a question:")

# Select embedding model
embed_model_name = st.selectbox(
    "Select Embedding Model for Candidate Retrieval",
    ("sentence-transformers/all-MiniLM-L6-v2", "nvidia/nv-embedqa-e5-v5")
)

# Load embedding model
st.write("Loading embedding model...")
embed_model = load_embedding_model(embed_model_name)

# Load dataset
st.write("Loading dataset...")
corpus, queries, qrels = load_dataset(num_docs=100)

# Ensure corpus is loaded
if corpus is None:
    st.write("Error: Corpus was not loaded correctly.")
else:
    # Proceed with retrieval if corpus is valid
    if query:
        st.write("Retrieving top-k passages...")
        try:
            top_k_passages = retrieve_top_k(embed_model, query, corpus, k=10)

            # Display retrieved passages
            st.write("Top-k passages before reranking:")
            for passage in top_k_passages:
                st.write(passage)

        except Exception as e:
            st.write(f"Error during retrieval: {e}")
