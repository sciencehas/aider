from transformers import AutoTokenizer

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer

def tokenize_data(tokenizer, text):
    input_ids = tokenizer.encode(text, add_special_tokens=True)
    return input_ids

def decode_data(tokenizer, input_ids):
    text = tokenizer.decode(input_ids, skip_special_tokens=True)
    return text

if __name__ == "__main__":
    model_name = "TheBloke/Luna-AI-Llama2-Uncensored-GGML"
    tokenizer = load_tokenizer(model_name)
    print("Tokenizer loaded successfully.")