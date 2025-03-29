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
