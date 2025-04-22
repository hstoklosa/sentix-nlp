import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def predict_sentiment(texts, model, tokenizer, batch_size=16):
    model.eval()
    predictions = []

    # Process data in batches
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size].tolist()

        # Tokenize
        encodings = tokenizer(batch_texts, truncation=True, padding=True, max_length=128, return_tensors="pt")
        input_ids = encodings["input_ids"].to(device)
        attention_mask = encodings["attention_mask"].to(device)

        # Predict
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(preds)

    return predictions


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
    