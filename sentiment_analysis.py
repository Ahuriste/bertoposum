# Load model directly
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
polarity_tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
polarity_model = AutoModelForSequenceClassification.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")



def get_polarity_batched (sentences,temperature=1):
    tokenized_sentence = polarity_tokenizer (sentences, padding=True, return_tensors="pt")
    logits = polarity_model(**tokenized_sentence).logits
    weights = torch.Tensor([-2/3,-1/3,0,1/3,2/3])
    probs = torch.sigmoid (logits/temperature)
    return torch.matmul(weights, probs.T).detach()



def get_polarity (sentences,temperature=1):
    tokenized_sentence = polarity_tokenizer (sentences, padding=True, return_tensors="pt")
    logits = polarity_model(**tokenized_sentence).logits
    weights = torch.Tensor([-2/3,-1/3,0,1/3,2/3])
    probs = torch.sigmoid (logits/temperature)
    return torch.dot(weights, probs[0]).detach()


