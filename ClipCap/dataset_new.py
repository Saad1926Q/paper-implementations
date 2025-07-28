from transformers import GPT2Tokenizer
from torch.utils.data import Dataset
from torch.linalg import vector_norm
import pickle
import torch
import os



class CaptionsDataset(Dataset):
    def __init__(self,clip_embeddings, captions,prefix_length:int,normalize_prefix:bool=False):
        super().__init__()
        self.tokenizer:GPT2Tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        self.prefix_length = prefix_length
        self.normalize_prefix = normalize_prefix

        assert len(clip_embeddings) == len(captions), "Embeddings and captions must be the same length."

        self.prefixes=clip_embeddings
        self.captions_tokens = []
        self.max_seq_len = 0

        for cap in captions:
            tokens = torch.tensor(self.tokenizer.encode(cap), dtype=torch.int64)
            self.captions_tokens.append(tokens)
            self.max_seq_len = max(self.max_seq_len, len(tokens))

        self.max_seq_len = min(64, self.max_seq_len) ## Just to ensure that the max seq len doesnt become too much
    
    
    def __len__(self) -> int:
        return len(self.captions_tokens)
    

    def pad_or_truncate_tokens(self,idx:int):
        tokens=self.captions_tokens[idx]
        padding=self.max_seq_len-len(tokens)

        if padding>0:
            tokens = torch.cat((tokens,torch.full((padding,),-1,dtype=torch.int64)))

        elif padding<0:
            # If sequence is longer than max_seq_len then we will have to truncate
            tokens = tokens[:self.max_seq_len]
            self.captions_tokens[idx] = tokens

        mask=torch.ge(tokens,0)
        tokens[~mask] = 0
        mask=mask.float()
        prefix_mask=torch.ones(self.prefix_length,dtype=torch.float32,device=tokens.device)
        full_mask = torch.cat((prefix_mask,mask))
        
        return tokens,full_mask
    
    def __getitem__(self, idx: int):
        tokens, mask = self.pad_or_truncate_tokens(idx)
        prefix = self.prefixes[idx]

        if self.normalize_prefix:
            prefix = prefix.float()
            prefix = prefix / vector_norm(prefix, ord=2)

        return tokens, mask, prefix