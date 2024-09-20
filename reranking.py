from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def load_ranking_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def rerank(model, tokenizer, query, top_k_passages):
    inputs = tokenizer([f"{query} [SEP] {passage}" for passage, _ in top_k_passages], return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs).logits
    scores = outputs.squeeze(-1)
    
    ranked_passages = sorted(zip(top_k_passages, scores), key=lambda x: x[1], reverse=True)
    return [(passage, score.item()) for (passage, _), score in ranked_passages]
