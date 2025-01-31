import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class MLMDataset(Dataset):
    def __init__(self, csv_path, tokenizer, text_column, max_length=4096, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.data = pd.read_csv(csv_path)[text_column].dropna().tolist()  # Read text column from CSV

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        
        input_ids = encoding["input_ids"].squeeze() 
        attention_mask = encoding["attention_mask"].squeeze()

        masked_input_ids = input_ids.clone()
        mask_labels = torch.full(input_ids.shape, -100)

        mask_prob = torch.full(input_ids.shape, self.mask_prob)
        masked_indices = torch.bernoulli(mask_prob).bool()

        mask_80 = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        masked_input_ids[mask_80] = tokenizer.mask_token_id 

        mask_10 = torch.bernoulli(torch.full(input_ids.shape, 0.1)).bool() & masked_indices
        random_tokens = torch.randint(0, tokenizer.vocab_size, input_ids.shape)
        masked_input_ids[mask_10] = random_tokens[mask_10]

        mask_labels[masked_indices] = input_ids[masked_indices]

        return {
            "masked_input_ids": masked_input_ids,
            "original_input_ids": input_ids,
            "mask_labels": mask_labels,
            "attention_mask": attention_mask
        }



def create_dataloaders(resume_path, jd_path, batch_size, tokenizer):
    resume_dataset = MLMDataset(resume_path, tokenizer, 'Resume_str')
    jd_dataset = MLMDataset(jd_path, tokenizer, 'job_description')
    resume_dataloader = DataLoader(resume_dataset, batch_size=batch_size, shuffle=True)
    jd_dataloader = DataLoader(jd_dataset, batch_size=batch_size, shuffle=True)
    return resume_dataloader, jd_dataloader
    

