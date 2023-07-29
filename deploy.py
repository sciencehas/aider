```python
from transformers import AutoModel
from flask import Flask, request, jsonify
import torch
from tokenizer import Tokenizer
from model import Model

app = Flask(__name__)

# Load the model
model = AutoModel.from_pretrained("TheBloke/Luna-AI-Llama2-Uncensored-GGML")
model.eval()

# Load the tokenizer
tokenizer = Tokenizer()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']

    # Tokenize the input text
    input_ids = tokenizer.encode(text)

    # Convert to tensor
    input_ids = torch.tensor([input_ids])

    # Get the model's prediction
    with torch.no_grad():
        outputs = model(input_ids)

    # Get the predicted token ids
    predicted_token_ids = torch.argmax(outputs.logits, dim=-1)

    # Decode the token ids to get the predicted text
    predicted_text = tokenizer.decode(predicted_token_ids[0])

    return jsonify({'predicted_text': predicted_text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```