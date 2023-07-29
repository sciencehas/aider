```python
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from model import model
from tokenizer import tokenizer
from data_processing import process_data

def train_model(train_data, val_data, model, tokenizer, epochs=5, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.train()

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=1e-5)

    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        avg_train_loss = total_loss / len(train_loader)
        print(f'Training loss for epoch {epoch+1}: {avg_train_loss}')

        model.eval()
        val_loss = 0
        for batch in val_loader:
            with torch.no_grad():
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation loss for epoch {epoch+1}: {avg_val_loss}')

        model.train()

    return model

if __name__ == "__main__":
    train_data, val_data = process_data('data/train.txt', 'data/val.txt', tokenizer)
    model = train_model(train_data, val_data, model, tokenizer)
    torch.save(model.state_dict(), 'model.pt')
```