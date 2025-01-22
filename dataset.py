import torch
from torch.utils.data import Dataset
import json

class TextLabelDataset(Dataset):
    def __init__(self, file_path = 'bags_and_cases_all.asp', general_aspect_id = 4, keep_all = False, num_label=9):
        """
        Args:
            file_path (str): Path to the file containing sentences and labels.
        """
        self.label_count = [0]*num_label
        self.data = []
        self.num_label = num_label
        with open(file_path, 'r') as file:
            for line in file:
                # Ignore lines containing product codes like B000...
                if line.strip() and not line.startswith("B"):
                    # Split line into text and label
                    parts = line.rsplit("\t", maxsplit=1)
                    if len(parts) == 2:
                        text, labels = parts
                        try:
                            # Convert label to an integer
                            labels = [int(label.strip()) for label in labels.split(" ")]
                            for i in labels:
                                self.label_count[i]+=1 
                            if keep_all or general_aspect_id not in labels:
                                self.data.append((text.strip(), labels))
                        except ValueError:
                            # Skip lines with non-integer labels
                            continue
        max_label_count = max(self.label_count)
        self.general_label = [i for i in range(self.num_label) if self.label_count[i]==max_label_count][0]
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        text, labels = self.data[idx]
        return text, labels

class OposumReviews(Dataset):
    def __init__(self, file_path = 'dev.json'):
        """
        Args:
            file_path (str): Path to the file containing sentences and labels.
        """
        self.data = []
        with open(file_path, 'r') as f:
            self.products = json.loads(f.read())

    def __len__(self):
        return len(self.products)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        product = self.products[idx]
        return [sentence for review in product["reviews"] for sentence in review["sentences"] ],  [summary for summary in product["summaries"]["general"]]
                                                            

