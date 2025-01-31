
from transformers import LongformerTokenizer, LongformerModel

def load_tokenizer_and_model(model_name):
    if model_name == "longformer-base-4096":
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096") 
        model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
        return tokenizer, model
    elif model_name == "something else":
        tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096") 
        model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
    else:
        raise ValueError("Unsupported model type.")