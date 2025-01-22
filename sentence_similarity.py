from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def compute_cosine_similarity(sentences):
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")

    with torch.no_grad():
        model_output = model(**encoded_input)
    attention_mask = encoded_input["attention_mask"]
    sentence_embeddings = mean_pooling(model_output, attention_mask)
    sentence_embeddings = sentence_embeddings.cpu().numpy()
    similarity_matrix = cosine_similarity(sentence_embeddings)

    return similarity_matrix

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
