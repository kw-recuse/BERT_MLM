
from transformers import AutoTokenizer, AutoModelForMaskedLM

def load_tokenizer_and_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name) 
    model = AutoModelForMaskedLM.from_pretrained(model_name)
    return tokenizer, model