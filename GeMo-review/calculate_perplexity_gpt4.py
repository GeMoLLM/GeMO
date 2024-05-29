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
parser.add_argument('--T_list', type=str, default='1.2')
parser.add_argument('--P_list', type=str, default='1.0')
parser.add_argument('--mode', type=str, default='personalized')
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--indices', type=str, default='')
parser.add_argument('--bid', type=int, default=-1)
parser.add_argument('--eid', type=int, default=-1)
parser.add_argument('--device', '-d', type=int, default=0)
args = parser.parse_args()

os.environ['TRANSFORMERS_CACHE'] = '../cache/huggingface/'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", low_cpu_mem_usage=True, torch_dtype="auto").cuda()

T_list = args.T_list.split(',')
P_list = args.P_list.split(',')

if args.bid != -1:
    indices = list(range(args.bid, args.eid))
else:
    indices = [int(x) for x in args.indices.split(',')]

# Encode the input text
folder = 'perplexity_scores/'

for i in tqdm(indices):
    for T in T_list:
        for P in P_list:
            perplexity_list = []

            data_path = f'goodreads_completions_{args.mode}_{args.model}-chat_500_temp-{T}-p-{P}-{i}_merged.json'
                    
            if not osp.exists(data_path):
                print('-------------------------missing\n-----', data_path)
                continue

            output_path = f'perplexity_goodreads_completions_{args.mode}_{args.model}-chat_500_temp-{T}-p-{P}-{i}_merged.npy'

            if osp.exists(osp.join(folder, output_path)):
                print('-------------------------exists\n-----', output_path)
                continue

            texts = json.load(open(data_path))
            for text in tqdm(texts):
                if not text:
                    perplexity_list.append(-1)
                    continue
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
                
            os.makedirs(folder, exist_ok=True)
            np.save(osp.join(folder, output_path), perplexity_list)