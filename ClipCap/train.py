from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from dataset_new import CaptionsDataset
import torch.nn.functional as F
from model import ClipCapModel
from torch.optim import AdamW
from tqdm import tqdm
import argparse
import pickle
import torch
import wandb
import json
import sys
import os


def train_model(dataset:CaptionsDataset,val_dataset: CaptionsDataset,model:ClipCapModel,args,lr:float=2e-5, warmup_steps: int = 5000, output_dir: str = ".", output_prefix: str = ""):
    run = wandb.init(
    entity="saad-ahmed1926q-jmi",
    project="clip-cap",
    config={
        "learning_rate": 2e-5,
        "architecture": "CLIP+MLP+GPT2",
        "dataset": "google-research-datasets/conceptual_captions",
        "epochs": 10,
    },
)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = args.bs
    epochs = args.epochs

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    model = model.to(device)
    model.train()
    optimizer = AdamW(model.parameters(), lr=lr)

    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=epochs * len(train_dataloader)
    )

    scaler = GradScaler(device='cuda')

    best_val_loss = float('inf')
    history = {"train_loss": [], "val_loss": []}
    
    for epoch in range(epochs):
        print(f"\t Training epoch {epoch}")
        sys.stdout.flush()

        model.train()
        progress = tqdm(total=len(train_dataloader), desc=output_prefix)

        train_loss_total = 0

        for idx, (tokens, mask, prefix) in enumerate(train_dataloader):
            tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
            
            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(tokens, prefix, mask)

                # you only want to train the model to predict caption tokens, not the prefix.

                # [batch_size, seq_len, vocab_size]

                # tensor[a:b] is shorthand for tensor[a:b, ...] — it keeps the remaining dimensions unchanged.

                logits = outputs.logits[:, dataset.prefix_length - 1: -1] 

                # We will reshape the logits to [B*T, V]
                # Each row is a distribution over the vocabulary 

                # Targets shape: [B*T]
                # Each entry is a single integer, the actual next token's index in the vocabulary.

                loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),tokens.flatten(),ignore_index=0) 
                

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()


            loss_cpu = loss.detach().cpu().item()
            train_loss_total += loss_cpu



            run.log({"train_step_loss": loss_cpu})
            progress.set_postfix({"loss": loss_cpu})
            progress.update()

            if (idx + 1) % 10000 == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(output_dir, f"{output_prefix}_latest.pt"),
                )

        progress.close()
        avg_train_loss = train_loss_total / len(train_dataloader)
        history["train_loss"].append(avg_train_loss)

        
        
        model.eval()
        val_loss_total = 0
        with torch.no_grad():
            for tokens, mask, prefix in val_dataloader:
                tokens, mask, prefix = tokens.to(device), mask.to(device), prefix.to(device, dtype=torch.float32)
                with autocast(device_type='cuda'):
                    outputs = model(tokens, prefix, mask)
                    logits = outputs.logits[:, val_dataset.prefix_length - 1: -1]
                    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.flatten(), ignore_index=0)
                    val_loss_total += loss.item()

        avg_val_loss = val_loss_total / len(val_dataloader)
        history["val_loss"].append(avg_val_loss)

        print(f"\tEpoch {epoch}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        run.log({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "lr": scheduler.get_last_lr()[0]
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f"{output_prefix}_best.pt"))

        torch.save(model.state_dict(), os.path.join(output_dir, f"{output_prefix}_latest.pt"))


    torch.save(history, os.path.join(output_dir, f"{output_prefix}_loss_history.pt"))

    run.finish()
    return model

