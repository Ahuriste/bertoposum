from tqdm import trange, tqdm
import os
from dataset import TextLabelDataset
from transformers import (
    RobertaTokenizerFast,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    AutoConfig,
)
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from random import sample
from dataset import TextLabelDataset, OposumReviews
from sentiment_analysis import get_polarity, get_polarity_batched
from evaluate import load
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from torch.optim import AdamW
from sentence_similarity import compute_cosine_similarity
import pandas as pd

from sklearn.metrics import f1_score, precision_score, recall_score

GRADIENT_ACCUMULATION_STEPS = 1

NUM_TRAIN_EPOCHS = 20
LEARNING_RATE = 1e-5
MAX_GRAD_NORM = 15
BATCH_SIZE = 4

first = True

def classify(sentences, temperature=1):

    model.eval()
    encoding = tokenizer(sentences,
                         padding=True,
                         truncation=True,
                         return_tensors="pt")
    encoding.to(device)
    logits = model(**encoding).logits
    labels = logits >= 0
    return logits, torch.sigmoid(logits / temperature).detach()


def evaluate_accuracy(model, dataloader, num_labels=9):
    model.eval()

    eval_loss = 0
    nb_eval_steps = 0
    all_predicted_labels, all_correct_labels = [], []

    for step, batch in enumerate(tqdm(dataloader, desc="Evaluation iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids = batch

        with torch.no_grad():
            out = model(input_ids, attention_mask=input_mask, labels=label_ids)

            logits = out.logits  # Shape: (B, L)
            tmp_eval_loss = out.loss

        # Convert logits to predictions (binarize)
        predicted_labels = (logits > 0).int().cpu().numpy()
        correct_labels = label_ids.cpu().numpy()
        all_predicted_labels.append(predicted_labels)
        all_correct_labels.append(correct_labels)
        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    all_predicted_labels = np.vstack(all_predicted_labels)
    all_correct_labels = np.vstack(all_correct_labels)

    # Compute metrics using sklearn
    macro_f1 = f1_score(all_correct_labels, all_predicted_labels, average="macro", zero_division=0)
    micro_f1 = f1_score(all_correct_labels, all_predicted_labels, average="micro", zero_division=0)
    weighted_f1 = f1_score(all_correct_labels, all_predicted_labels, average="weighted", zero_division=0)

    macro_precision = precision_score(all_correct_labels, all_predicted_labels, average="macro", zero_division=0)
    macro_recall = recall_score(all_correct_labels, all_predicted_labels, average="macro", zero_division=0)

    # Return metrics
    return {
        "eval_loss": eval_loss,
        "macro_f1": macro_f1,
        "micro_f1": micro_f1,
        "weighted_f1": weighted_f1,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
    }



"""
def evaluate_accuracy_(model, dataloader, num=4):
    model.eval()

    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []
    precision = recall = 0

    for step, batch in enumerate(tqdm(dataloader,
                                      desc="Evaluation iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids = batch
        with torch.no_grad():
            out = model(input_ids, attention_mask=input_mask, labels=label_ids)

            logits = out.logits
            tmp_eval_loss = out.loss
        outputs = np.argmax(logits.to('cpu'), axis=1)
        label_ids = label_ids.to('cpu').numpy()
        tp = sum([
            logits[i][j] > 0 and label_ids[i][j] == 1
            for i in range(len(logits)) for j in range(num_labels)
        ])
        recall += (tp / label_ids.sum()).item()
        if (logits > 0).sum() != 0:
            precision += tp / (logits > 0).sum().item()
        #predicted_labels += list(outputs)
        #correct_labels += list(label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps

    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
    recall /= len(dataloader)
    precision /= len(dataloader)
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = ((2 * precision * recall) / (precision + recall)).cpu().item()
    return eval_loss, f1


def evaluate_accuracy(model, dataloader, num=4, num_labels=9):
    model.eval()

    eval_loss = 0
    nb_eval_steps = 0
    predicted_labels, correct_labels = [], []
    precision = recall = 0

    tp = torch.zeros(num_labels)
    p = torch.zeros(num_labels)
    pp = torch.zeros(num_labels)

    for step, batch in enumerate(tqdm(dataloader,
                                      desc="Evaluation iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, label_ids = batch
        with torch.no_grad():
            out = model(input_ids, attention_mask=input_mask, labels=label_ids)

            logits = out.logits
            tmp_eval_loss = out.loss
        outputs = np.argmax(logits.to('cpu'), axis=1)
        label_ids = label_ids.to('cpu')
        predicted = (logits >= 0).to('cpu')

        tp += ((predicted & (label_ids != 0)).sum(dim=0))
        p += label_ids.sum(dim=0)
        pp += predicted.sum(dim=0)
        tp = sum([logits[i][j]>0 and label_ids[i][j]==1 for i in range(len(logits) )for j in range(num_labels)])
        recall += (tp/label_ids.sum()).item()
        if (logits>0).sum() != 0:
            precision += tp/(logits>0).sum().item()
        #predicted_labels += list(outputs)
        #correct_labels += list(label_ids)

        eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps

    correct_labels = np.array(correct_labels)
    predicted_labels = np.array(predicted_labels)
    #recall /= len(dataloader)
    #precision /= len(dataloader)
    precision = [tp[i] / pp[i] if pp[i] else 0 for i in range(num_labels)]
    recall = [tp[i] / p[i] if p[i] else 0 for i in range(num_labels)]
    f1 = []
    for i in range(num_labels):
        if precision[i] + recall[i] == 0:
            f1.append(0)
        else:
            f1.append(((2 * precision[i] * recall[i]) /
                       (precision[i] + recall[i])).cpu().item())
    return eval_loss, sum(f1) / num_labels
    """


def collate_fn(batch):
    texts, labels = zip(*batch)
    encoding = tokenizer(list(texts),
                         padding=True,
                         truncation=True,
                         return_tensors="pt")

    labels = torch.stack([
        torch.nn.functional.one_hot(torch.tensor(index),
                                    num_classes=num_labels).sum(dim=0)
        for index in labels
    ]).float()
    #labels = torch.nn.functional.one_hot(torch.tensor(labels),num_classes=num_labels)
    return encoding['input_ids'], encoding["attention_mask"], labels


metric_names = ["rouge1", "rouge2", "rougeL"]
pipes = ["P", "A", "A+P"]
metrics = {m: pd.DataFrame({p: [] for p in pipes}) for m in metric_names}
if not first:
    for key in metrics.keys():
        metrics[key] = pd.read_csv(f"{key}.csv",index_col=0)

losses_df = pd.DataFrame({"f1": []})
#if not first:
#    losses_df = pd.read_csv("loss.csv",index_col=0)
model_name = "roberta-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_labels = 9
domain_codes_asp = [
    "bluetooth", "boots", "bags_and_cases", "vacuums", "tv", "keyboards"
]
domain_codes_json = ["bt", "boots", "bag", "vacuum", "tv", "keyboard"]
for counter in range(20):
    print(counter)
    for domain_code_asp, domain_code_json in zip(domain_codes_asp,
                                                domain_codes_json):
        print(domain_code_asp)
        model = RobertaForSequenceClassification.from_pretrained(
            model_name,
            problem_type="multi_label_classification",
            num_labels=num_labels)

        model.to(device)

        tokenizer = RobertaTokenizerFast.from_pretrained(model_name)

        #model.load_state_dict (torch.load ("models/pytorch_model_bags.bin", map_location=device) )

        ### TO CHANGE ###
        dev_dataset = TextLabelDataset(file_path=f"data/{domain_code_asp}-dev.asp",
                                    keep_all=True)
        test_dataset = TextLabelDataset(
            file_path=f"data/{domain_code_asp}-tst.asp", keep_all=True)

        train_size = int(0.75 * len(dev_dataset))
        val_size = len(dev_dataset) - train_size
        train_dataset, val_dataset = random_split(dev_dataset,
                                                [train_size, val_size])
        ##################

        general_aspect_label = dev_dataset.general_label


        train_dataloader = DataLoader(train_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=True,
                                    collate_fn=collate_fn)
        test_dataloader = DataLoader(test_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    collate_fn=collate_fn)
        val_dataloader = DataLoader(val_dataset,
                                    batch_size=BATCH_SIZE,
                                    shuffle=False,
                                    collate_fn=collate_fn)

        num_train_steps = int(
            len(train_dataloader.dataset) / BATCH_SIZE /
            GRADIENT_ACCUMULATION_STEPS * NUM_TRAIN_EPOCHS)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [{
            'params':
            [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay':
            0.01
        }, {
            'params':
            [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay':
            0.0
        }]

        optimizer = AdamW(optimizer_grouped_parameters, lr=LEARNING_RATE)
        #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=num_warmup_steps, t_total=num_train_steps)

        OUTPUT_DIR = "models/"
        MODEL_FILE_NAME = "pytorch_model.bin"
        PATIENCE = 16

        loss_history = []
        no_improvement = 0
        for _ in trange(int(NUM_TRAIN_EPOCHS), desc="Epoch"):
            model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            for step, batch in enumerate(
                    tqdm(train_dataloader, desc="Training iteration")):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, label_ids = batch

                outputs = model(input_ids,
                                attention_mask=input_mask,
                                labels=label_ids)
                loss = outputs.loss

                if GRADIENT_ACCUMULATION_STEPS > 1:
                    loss = loss / GRADIENT_ACCUMULATION_STEPS

                loss.backward()
                tr_loss += loss.item()

                if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:

                    optimizer.step()
                    optimizer.zero_grad()
                    #scheduler.step()

            f1 = evaluate_accuracy(model, val_dataloader)["weighted_f1"]
            """print("Loss history:", loss_history)
                print("Dev loss:", dev_loss)
                print("Dev f1:", f1)
                """
            if len(loss_history) == 0 or f1 > max(loss_history):
                no_improvement = 0
                model_to_save = model.module if hasattr(model, 'module') else model
                output_model_file = os.path.join(OUTPUT_DIR, MODEL_FILE_NAME)
                torch.save(model_to_save.state_dict(), output_model_file)
                print(_)
                print(f1)
            else:
                no_improvement += 1

            if no_improvement >= PATIENCE:
                #print("No improvement on development set. Finish training.")
                break

            loss_history.append(f1)

        model.load_state_dict(
            torch.load("models/pytorch_model.bin", map_location=device))
        f1 = evaluate_accuracy(model, test_dataloader)["weighted_f1"]
        losses_df.loc[domain_code_asp, "f1"] = f1
        rouge = load("rouge")
        """
            CHANGE ME
            THIS NEXT LINE NEEDS TO BE CHANGED WHEN ON A DIFFERENT DOMAIN
            """
        for dataset_file in [f'data/oposum/{domain_code_json}/test.json']:
            oposum = OposumReviews(file_path=dataset_file)
            all_sorted_sentences = []
            predictions = []
            references = []
            coeffs = [(0, 1), (1, 0), (1, 1)]
            for pipe_index, coeff in enumerate(coeffs):
                for sentences, reviews in oposum:
                    pol = get_polarity_batched(sentences).cpu()
                    aspects = classify(sentences)
                    spikedness = (aspects[1].cpu().max(dim=1)[0] -
                                aspects[1].cpu()[:, general_aspect_label])
                    spikedness = spikedness / spikedness.mean()
                    salience = spikedness**(coeff[0]) * pol.abs()**coeff[1]
                    salience_ = sorted([(i, s) for i, s in enumerate(salience)],
                                    key=lambda x: x[1],
                                    reverse=True)
                    sorted_sentences = [sentences[i] for i, s in salience_]
                    all_sorted_sentences.append(sorted_sentences)
                    sim = compute_cosine_similarity(sorted_sentences)
                    prediction = ""
                    indexes = []
                    for i in range(6):
                        if prediction == "" or max([sim[i][j]
                                                    for j in indexes]) <= 0.45:
                            indexes.append(i)
                            prediction += sorted_sentences[i]

                        if len(indexes) >= 5:
                            break
                    predictions.append(prediction)
                    references.append(reviews)
                rouge_metrics = rouge.compute(references=references,
                                            predictions=predictions,
                                            use_stemmer=True)
                for metric in metric_names:
                    if first and counter == 0:
                        metrics[metric].loc[domain_code_asp,
                                        pipes[pipe_index]] = rouge_metrics[metric]
                    else:
                        metrics[metric].loc[domain_code_asp,
                                        pipes[pipe_index]] += rouge_metrics[metric]
    for key, value in metrics.items():
        value.to_csv(f"{key}.csv")
    losses_df.to_csv(f"loss{counter}.csv")
