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
parser.add_argument('--decay_list', type=str, default='linear,exponential')
parser.add_argument('--P_list', type=str, default='0.90,0.95,0.98,1.00')
parser.add_argument('--mode', type=str, default='personalized')
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--indices', type=str, default='')
parser.add_argument('--bid', type=int, default=-1)
parser.add_argument('--eid', type=int, default=-1)
parser.add_argument('--device', '-d', type=int, default=0)
parser.add_argument('--end_temperature', '-et', type=float, default=1.0)
parser.add_argument('--period', '-p', type=int, default=20)
args = parser.parse_args()

os.environ['TRANSFORMERS_CACHE'] = '../cache/huggingface/'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", low_cpu_mem_usage=True, torch_dtype="auto").cuda()

decay_list = args.decay_list.split(',')
P_list = args.P_list.split(',')

if args.bid != -1:
    indices = list(range(args.bid, args.eid))
else:
    indices = [int(x) for x in args.indices.split(',')]

# Encode the input text
folder = 'perplexity_scores/'
os.makedirs(folder, exist_ok=True)

for i in tqdm(indices):
    for T in decay_list:
        for P in P_list:
            perplexity_list = []
            data_path = f'goodreads_completions_{args.mode}_{args.model}-chat_500_decay-{T}_p-{P}_k-50-{i}_merged.json' \
                if args.end_temperature == 1.0 and args.period == 20 \
                    else f'goodreads_completions_{args.mode}_{args.model}-chat_500_decay-{T}-{args.end_temperature}-{args.period}_p-{P}_k-50-{i}_merged.json'
            output_path = f'perplexity_goodreads_completions_{args.mode}_{args.model}-chat_500_decay-{T}_p-{P}_k-50-{i}_merged.npy' \
                if args.end_temperature == 1.0 and args.period == 20 \
                    else f'perplexity_goodreads_completions_{args.mode}_{args.model}-chat_500_decay-{T}-{args.end_temperature}-{args.period}_p-{P}_k-50-{i}_merged.npy'

            if not osp.exists(data_path):
                print(f'{data_path} does not exist')
                continue
            
            if osp.exists(osp.join(folder, output_path)):
                print(f'{output_path} already exists. skipping...')
                continue

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