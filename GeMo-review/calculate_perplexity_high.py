import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import os.path as osp
import json
from torch.nn.functional import cross_entropy
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--prefix', type=str, default='')
parser.add_argument('--n', type=int, default=5)
parser.add_argument('--device', '-d', type=int, default=0)
args = parser.parse_args()

os.environ['TRANSFORMERS_CACHE'] = '../cache/huggingface/'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", low_cpu_mem_usage=True, torch_dtype="auto").cuda()

# Encode the input text
folder = 'perplexity_scores/'
os.makedirs(folder, exist_ok=True)

for i in range(args.n):
    data_path = f'{args.prefix}-{i}_merged.json'
    output_path = f'perplexity_{args.prefix}-{i}_merged.npy'
    perplexity_list = []
    texts = json.load(open(data_path))
    for text in tqdm(texts):
        input_ids = tokenizer.encode(text, return_tensors="pt").cuda()
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
            
        # Calculate cross-entropy loss
        # Shift the logits and labels so that they align (the model predicts the next token)
        shifted_logits = logits[:, :-1, :]
        shifted_labels = input_ids[:, 1:]
        loss = cross_entropy(shifted_logits.view(-1, shifted_logits.size(-1)), shifted_labels.view(-1), reduction='mean')
        
        perplexity = torch.exp(loss)
        
        perplexity_list.append(perplexity.item())
        
        del input_ids, outputs, logits, shifted_logits, shifted_labels, loss, perplexity
        
    np.save(osp.join(folder, output_path), perplexity_list)