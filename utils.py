import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(preds, labels, average='weighted'):
    """
    Compute accuracy, precision, recall, and F1 score based on predictions.   
    """
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average=average
    )
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

class NewsDataset(torch.utils.data.Dataset):
    """
    Create a PyTorch dataset for text classification using BERT models.
    """
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts.iloc[idx]
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].flatten(),
            'attention_mask': inputs['attention_mask'].flatten(),
            'label': torch.tensor(self.labels.iloc[idx], dtype=torch.long)
        }
    