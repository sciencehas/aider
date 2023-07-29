```python
from transformers import AutoTokenizer
import torch

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("TheBloke/Luna-AI-Llama2-Uncensored-GGML")

def process_data(data):
    """
    Function to process the data and return it in a format suitable for the model.
    """
    # Tokenize the data
    tokenized_data = tokenizer(data, truncation=True, padding=True, return_tensors="pt")

    # Convert the labels to categorical values
    labels = torch.tensor([0 if label == 'negative' else 1 for label in data['labels']])

    return tokenized_data, labels
```