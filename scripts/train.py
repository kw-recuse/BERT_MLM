import os 
import json
from tqdm.notebook import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.nn import functional as F
from models.base import load_tokenizer_and_model
from data.dataloader import create_dataloaders

class Trainer:
    def __init__(self, config_file, **kwargs):
        self.config = self._load_config(config_file)
        
        # take the keyword args
        for key in ['checkpoints_path', 'csv_file_paths', 'col_names']:
            if key in kwargs:
                self.config[key] = kwargs[key]
        
        self.model_name = self.config['model_name']
        self.device = self.config['device']
        self.batch_size = self.config['batch_size']
        self.checkpoints_path = self.config['checkpoints_path']
        self.lr = self.config['lr']
        self.epoch_num = self.config['epoch_num']
        self.csv_file_paths = self.config['csv_file_paths']
        self.col_names = self.config['col_names']
        self.log_per_epoch = self.config['log_per_epoch']
        self.use_fp16 = self.config['use_fp16']
        self.max_length = self.config['max_length']
        
        # create a checkpoint path
        os.makedirs(self.checkpoints_path, exist_ok=True)
    
        # initalize tokenizer and model
        self.tokenizer, self.model = load_tokenizer_and_model(self.model_name)
        
        self.model = self.model.to(self.device)
        
        # define optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
        
        # initalize data loaders
        self.train_dataloader, self.val_dataloader = create_dataloaders(self.csv_file_paths, self.col_names,  self.tokenizer, self.batch_size)
        
        self.log_step = len(self.train_dataloader) // self.log_per_epoch
        
        # set scaler
        self.scaler = GradScaler()
        
        # initalize list to hold train losses, and val losses
        self.train_losses = []
        self.val_losses = []
        
    @staticmethod
    def _load_config(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
        
    @staticmethod
    def mlm_loss(predictions, labels):
        mask_positions = labels != -100
        pred_masked = predictions[mask_positions]
        labels_masked = labels[mask_positions]
        loss = F.cross_entropy(pred_masked, labels_masked)
        return loss
    
    def evaluate_val_loss(self, epoch, step):
        self.model.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.no_grad():
            for val_batch in self.val_dataloader:
                val_masked_input_ids = val_batch["masked_input_ids"].to(self.device)
                val_mask_labels = val_batch["mask_labels"].to(self.device)
                val_attention_mask = val_batch["attention_mask"].to(self.device)
                
                if self.use_fp16:
                    with torch.amp.autocast(device_type='cuda'):
                        val_outputs = self.model(val_masked_input_ids, attention_mask=val_attention_mask)
                        val_logits = val_outputs.logits
                        val_loss += self.mlm_loss(val_logits.view(-1, val_logits.size(-1)), val_mask_labels.view(-1)).item()
                else:
                    val_outputs = self.model(val_masked_input_ids, attention_mask=val_attention_mask)
                    val_logits = val_outputs.logits
                    val_loss += self.mlm_loss(val_logits.view(-1, val_logits.size(-1)), val_mask_labels.view(-1)).item()
                    
                val_steps += 1
                
        avg_val_loss = val_loss / val_steps
        avg_val_loss = round(avg_val_loss, 4)
        self.val_losses.append(avg_val_loss)
        tqdm.write(f"Validation Loss at epoch: {epoch} - step {step}: {avg_val_loss}")
        self.model.train()
        return avg_val_loss
    
    def save_checkpoint(self, epoch, step, val_loss):
        checkpoint_path = os.path.join(self.checkpoints_path, f"epoch{epoch}_step{step}_loss{val_loss}.pt")
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict()
        }, checkpoint_path)
        tqdm.write(f"Checkpoint saved at {checkpoint_path}")
        
        
    def train(self):
        self.model.train()
        for epoch in range(self.epoch_num):
            progress_bar = tqdm(enumerate(self.train_dataloader), 
                                total=len(self.train_dataloader), 
                                desc=f"Epoch {epoch+1}", position=0, leave=True)
            
            for step, batch in progress_bar:
                masked_input_ids = batch["masked_input_ids"].to(self.device)
                mask_labels = batch["mask_labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                
                # add fp16 options
                if self.use_fp16:
                    with torch.amp.autocast(device_type='cuda'):
                        outputs = self.model(masked_input_ids, attention_mask=attention_mask)  
                        logits = outputs.logits
                        loss = self.mlm_loss(logits.view(-1, logits.size(-1)), mask_labels.view(-1))
                else:
                    outputs = self.model(masked_input_ids, attention_mask=attention_mask)  
                    logits = outputs.logits
                    loss = self.mlm_loss(logits.view(-1, logits.size(-1)), mask_labels.view(-1))
                
                self.optimizer.zero_grad()
                
                if self.use_fp16:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                
                self.train_losses.append(round(loss.item(), 4))
                progress_bar.set_postfix(Step=step+1, Loss=round(loss.item(), 4))
                
                # get the loss on validation set and save checkpoint
                if (step+1) % self.log_step == 0 or step == len(self.train_dataloader) - 1:
                    val_loss = self.evaluate_val_loss(epoch+1, step+1) # get the val loss
                    self.save_checkpoint(epoch+1, step+1, val_loss) # save the checkpoint