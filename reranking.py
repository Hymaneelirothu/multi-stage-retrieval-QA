from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

def load_ranking_model(model_name):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

def rerank(model, tokenizer, query, passages):
    inputs = tokenizer([query] * len(passages), passages, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    scores = torch.softmax(logits, dim=1)[:, 1]  # Assuming the positive class is at index 1
    ranked_passages = [passages[i] for i in scores.argsort(descending=True)]
    return ranked_passages
