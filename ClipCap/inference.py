import torch
import os
import cv2
import argparse
import requests
import numpy as np
from PIL import Image
from io import BytesIO
from torch.optim import AdamW
from model import ClipCapModel
from torchvision import transforms
from dataset_new import CaptionsDataset
from inference_utils import greedy_decoding,top_k_sampling
from transformers import GPT2Tokenizer, CLIPProcessor, CLIPModel,GPT2LMHeadModel

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path: str, prefix_length=10, clip_length=10, prefix_size=512):
    model = ClipCapModel(prefix_length=prefix_length, clip_length=clip_length, prefix_size=prefix_size)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    if os.path.exists(checkpoint_path):
        print(f"Loading weights from {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    
    return model

def preprocess_image_from_url(image_url):
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")
    return image 


def get_clip_embedding(pil_image):
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_model.eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    inputs = processor(images=[pil_image], return_tensors="pt").to(device)

    with torch.no_grad():
        embedding = clip_model.get_image_features(**inputs)


    return embedding




def generate_caption(model, clip_embedding, tokenizer: GPT2Tokenizer, method='greedy', max_length=12):
    model.eval()
    with torch.no_grad():
        prefix = model.mapping(clip_embedding).view(-1, model.prefix_length, model.gpt_embedding_size)

        if method == 'greedy':
            generated_ids = greedy_decoding(max_length, prefix, model, tokenizer)
        elif method == 'topk':
            generated_ids = top_k_sampling(max_length, prefix, model, tokenizer)
        else:
            raise ValueError(f"Unsupported decoding method: {method}")

        output_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return output_text.strip()

    

model=load_model("./checkpoints/coco_prefix_best_200k.pt")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

parser = argparse.ArgumentParser()
parser.add_argument('--url',type=str,default="https://t4.ftcdn.net/jpg/03/24/42/21/360_F_324422176_Lgn7NTeFyNaUKIDu0Ppls1u8zb8wsKS4.jpg" )
parser.add_argument('--method', type=str, default='greedy', choices=['greedy', 'topk'], help="Decoding method: 'greedy' or 'topk'")

args = parser.parse_args()

image_url = args.url

image = preprocess_image_from_url(image_url)

clip_embedding = get_clip_embedding(image)

caption = generate_caption(model, clip_embedding, tokenizer,method=args.method)
print("Generated Caption:", caption)

