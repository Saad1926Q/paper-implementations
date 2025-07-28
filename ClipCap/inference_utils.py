import torch
import torch.nn.functional as F

def greedy_decoding(max_length,prefix,model,tokenizer):
    generated_ids = None

    for i in range(max_length):

        if generated_ids is not None:
            token_embeddings=model.gpt.transformer.wte(generated_ids)
            full_embeddings = torch.cat((prefix, token_embeddings), dim=1)
        else:
            full_embeddings=prefix

        output=model.gpt(inputs_embeds=full_embeddings)

        logits=output.logits   #(1 , sequence_length, vocab_size)

        logits_next = logits[:, -1, :]  # (1, vocab_size)

        
        next_token_id = torch.argmax(logits_next, dim=-1).unsqueeze(0)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        if generated_ids is None:
            generated_ids = next_token_id
        else:
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

    return generated_ids





def top_k_sampling(max_length,prefix,model,tokenizer,k=3):
    generated_ids = None

    for i in range(max_length):

        if generated_ids is not None:
            token_embeddings=model.gpt.transformer.wte(generated_ids)
            full_embeddings = torch.cat((prefix, token_embeddings), dim=1)
        else:
            full_embeddings=prefix

        output=model.gpt(inputs_embeds=full_embeddings)

        logits=output.logits   #(1 , sequence_length, vocab_size)

        logits_next = logits[:, -1, :]  # (1, vocab_size)
        
        probs=F.softmax(logits_next,dim=1)

        topk,topk_idx=torch.topk(probs,k,dim=1)

        topk_idx=topk_idx.squeeze(0)

        normalized_probs=F.normalize(topk,p=1,dim=1).squeeze(0)

        sampled=torch.multinomial(normalized_probs,num_samples=1)

        next_token_id=topk_idx[sampled.item()].unsqueeze(0).unsqueeze(0)

        if next_token_id.item() == tokenizer.eos_token_id:
            break

        if generated_ids is None:
            generated_ids = next_token_id
        else:
            generated_ids = torch.cat((generated_ids, next_token_id), dim=1)

    return generated_ids

