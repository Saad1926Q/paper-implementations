import torch
from typing import Optional,Tuple
from transformers import GPT2LMHeadModel 


class MLP(torch.nn.Module):
    def __init__(self,prefix_size,intermediate_size,out_size):
        super().__init__()
        layers=[]
        self.proj1=torch.nn.Linear(prefix_size,intermediate_size,bias=True)
        self.proj2=torch.nn.Linear(intermediate_size,out_size,bias=True)
    
    def forward(self,X):
        z=self.proj1(X)
        z=torch.nn.functional.tanh(z)
        z=self.proj2(z)
        return z


class ClipCapModel(torch.nn.Module):
    def __init__(self,prefix_length:int,clip_length:Optional[int]=None,prefix_size:int=512,num_layers:int=8):
        super().__init__()
        self.prefix_length=prefix_length
        self.gpt=GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]

        self.mapping=MLP(prefix_size=prefix_size,intermediate_size=(self.gpt_embedding_size * prefix_length) // 2,out_size=self.gpt_embedding_size * prefix_length)

    
    def forward(self,tokens:torch.Tensor,prefix:torch.Tensor,mask:Optional[torch.Tensor]=None,labels: Optional[torch.Tensor] = None):
        text_embeddings=self.gpt.transformer.wte(tokens) # word token embedding layer . Each token ID is turned into an embedding.
        prefix_mapped=self.mapping(prefix).view(-1,self.prefix_length,self.gpt_embedding_size) # Go from (batch_size,self.prefix_length * self.gpt_embedding_size) to (batch_size,self.prefix_length,self.gpt_embedding_size)

        embeddings=torch.cat((prefix_mapped,text_embeddings),dim=1)

        batch_size=tokens.shape[0]
        
        # For training, GPT-2 needs a label for every input token.

        if labels is not None:
            # insert dummy tokens (zeros) in the label for the prefix part since there’s no ground-truth text corresponding to that.
            dummy_tokens=torch.zeros(batch_size,self.prefix_length)
            labels=torch.cat((dummy_tokens,tokens),dim=1)

        out=self.gpt(inputs_embeds=embeddings, labels=labels, attention_mask=mask) 


        return out
    
