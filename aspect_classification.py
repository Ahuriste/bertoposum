import torch
from tqdm import tqdm
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split



model_name = "roberta-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 9
model = RobertaForSequenceClassification.from_pretrained(model_name, 
                                                           problem_type="multi_label_classification", 
                                                           num_labels=num_labels)

model.to(device)
tokenizer = RobertaTokenizerFast.from_pretrained(model_name)
    
model.load_state_dict (torch.load ("models/pytorch_model.bin", map_location=device) )


def classify (sentences,temperature=1):
    model.eval()
    encoding = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    encoding.to(device)
    logits = model(**encoding).logits
    labels = logits>=0
    return logits, torch.sigmoid(logits/temperature).detach()

