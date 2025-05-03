import torch
from torch.utils.data import random_split, Dataset, DataLoader, ConcatDataset

import pandas as pd

class MLMDataset(Dataset):
    def __init__(self, csv_path, tokenizer, text_column, max_length=512, mask_prob=0.15):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob = mask_prob
        self.data = pd.read_csv(csv_path)[text_column].dropna().tolist()
        
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

        # Clone for masking
        masked_input_ids = input_ids.clone()
        mask_labels = torch.full(input_ids.shape, -100) 

        # Apply Masking
        mask_prob = torch.full(input_ids.shape, self.mask_prob) 
        masked_indices = torch.bernoulli(mask_prob).bool()

        mask_80 = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        masked_input_ids[mask_80] = self.tokenizer.mask_token_id 

        mask_10 = torch.bernoulli(torch.full(input_ids.shape, 0.1)).bool() & masked_indices
        random_tokens = torch.randint(0, self.tokenizer.vocab_size, input_ids.shape)
        masked_input_ids[mask_10] = random_tokens[mask_10]

        mask_labels[masked_indices] = input_ids[masked_indices] 

        return {
            "masked_input_ids": masked_input_ids,
            "original_input_ids": input_ids,
            "mask_labels": mask_labels,
            "attention_mask": attention_mask
        }


def create_dataloaders(csv_file_paths, col_names, max_length, tokenizer, batch_size, train_ratio=0.95):
    if len(csv_file_paths) != len(col_names):
        raise ValueError(f"Expected the number of CSV file paths to equal the number of column names, but got {len(csv_file_paths)} and {len(col_names)}")
    
    datasets = []
    for csv_file, col_name in zip(csv_file_paths, col_names):
        datasets.append(MLMDataset(csv_file, tokenizer, col_name, max_length))
        
    combined_dataset = ConcatDataset(datasets)
    train_size = int(train_ratio * len(combined_dataset))
    val_size = len(combined_dataset) - train_size
    train_dataset, val_dataset = random_split(combined_dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    return train_dataloader, val_dataloader