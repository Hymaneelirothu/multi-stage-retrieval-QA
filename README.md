---

# Multi-Stage Retrieval Pipeline for Question Answering

This project implements a multi-stage text retrieval pipeline using embedding and ranking models for question-answering tasks. The application is built with Streamlit.

## Features

- **Stage 1**: Candidate retrieval using embedding models.
- **Stage 2**: Reranking the retrieved passages with ranking models.
- **Custom Evaluation**: Performance measured using custom metrics (no BEIR).

## Setup Instructions

### Prerequisites

- Python 3.8+ installed.

### Steps

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/Hymaneelirothu/multi-stage-retrieval-QA.git
   cd multi-stage-retrieval
   ```

2. **Install Dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Application**:

   ```bash
   streamlit run app.py
   ```

## Usage

- Input a question, and the system retrieves and reranks relevant passages from the dataset.
- Custom evaluation metrics like NDCG@10.

## Dataset

- Natural Questions (NQ) dataset from the BEIR benchmark.

