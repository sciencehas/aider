```python
import torch
from sklearn.metrics import accuracy_score
from aider_app.model import model
from aider_app.tokenizer import tokenizer
from aider_app.data_processing import process_data

def test_model(test_loader):
    model.eval()
    predictions, true_labels = [], []

    for batch in test_loader:
        inputs = tokenizer(batch["text"], padding=True, truncation=True, max_length=512, return_tensors="pt")
        inputs = inputs.to('cuda')
        labels = batch["label"].to('cuda')

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_label = torch.argmax(logits, dim=1)

        predictions.extend(predicted_label.tolist())
        true_labels.extend(labels.tolist())

    print(f"Test Accuracy: {accuracy_score(true_labels, predictions)}")

if __name__ == "__main__":
    test_data = process_data("test_data.csv")
    test_model(test_data)
```