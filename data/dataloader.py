import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd

class MLMDataset(Dataset):
    def __init__(self, csv_paths, col_names, tokenizer, max_length=512, mask_prob=0.15, drop_last=True, stride=None):
        if len(csv_paths) != len(col_names):
            raise ValueError("csv_paths and col_names length mismatch")

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prob  = mask_prob

        texts = []
        for p, c in zip(csv_paths, col_names):
            texts.extend(pd.read_csv(p)[c].dropna().tolist())
        big_text = " ".join(texts)

        token_ids = tokenizer.encode(big_text, add_special_tokens=False)

        if stride is None:
            stride = max_length                      
        blocks = []
        for i in range(0, len(token_ids), stride):
            chunk = token_ids[i:i + max_length]
            if len(chunk) < max_length and drop_last:
                break
            if len(chunk) < max_length:
                chunk += [tokenizer.pad_token_id] * (max_length - len(chunk))
            blocks.append(torch.tensor(chunk, dtype=torch.long))
        self.blocks = blocks

    def __len__(self):
        return len(self.blocks)

    def __getitem__(self, idx):
        input_ids = self.blocks[idx].clone()
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()

        labels = torch.full_like(input_ids, -100)
        masked = torch.rand_like(input_ids.float()) < self.mask_prob
        labels[masked] = input_ids[masked]

        # 80 % → [MASK]
        mask80 = masked & (torch.rand_like(input_ids.float()) < 0.8)
        input_ids[mask80] = self.tokenizer.mask_token_id

        # 10 % → random token
        mask10 = masked & ~mask80 & (torch.rand_like(input_ids.float()) < 0.5)
        rand_tokens = torch.randint(0, self.tokenizer.vocab_size, input_ids.shape)
        input_ids[mask10] = rand_tokens[mask10]
        
        
        # 10% -> Dont do anything

        return {
            "masked_input_ids": input_ids,
            "mask_labels": labels,
            "attention_mask": attention_mask,
        }


def get_loaders(csv_paths, col_names, tokenizer, batch_size, max_length=512, mask_prob=0.15, train_ratio=0.95, seed=42,):
    dataset = MLMDataset(csv_paths, col_names, tokenizer, max_length=max_length, mask_prob=mask_prob)
    train_len = int(train_ratio * len(dataset))
    val_len   = len(dataset) - train_len

    train_set, val_set = random_split(dataset, [train_len, val_len], generator=torch.Generator().manual_seed(seed))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader