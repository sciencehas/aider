```python
from transformers import AutoModel

def load_model():
    model_name = "TheBloke/Luna-AI-Llama2-Uncensored-GGML"
    model = AutoModel.from_pretrained(model_name)
    return model
```