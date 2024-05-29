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
parser.add_argument('--input_folder', type=str, default='')
parser.add_argument('--device', '-d', type=int, default=0)
args = parser.parse_args()

os.environ['TRANSFORMERS_CACHE'] = '../cache/huggingface/'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", low_cpu_mem_usage=True, torch_dtype="auto").cuda()

# Encode the input text
folder = 'perplexity_scores/'

files = os.listdir(args.input_folder)
d_perplexity = {}
for file in tqdm(files):
    file_path = osp.join(args.input_folder, file)
    perplexity_list = []
    with open(file_path) as f:
        for line in f:
            text = json.loads(line)['review_text']
            
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
            d_perplexity[file] = perplexity_list

            del input_ids, outputs, logits, shifted_logits, shifted_labels, loss, perplexity
                
os.makedirs(folder, exist_ok=True)
output_path = f'perplexity_goodreads_src.json'
json.dump(d_perplexity, open(osp.join(folder, output_path), 'w'))