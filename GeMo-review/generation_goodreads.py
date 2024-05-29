import argparse
import json
import transformers
from tqdm import tqdm
import torch
import numpy as np
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='meta-llama/Llama-2-13b-chat-hf')
parser.add_argument('--output_path', type=str, default='BOLD_completions_debug')
parser.add_argument('--max_new_tokens', type=int, default=500)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--top_k', type=int, default=50)
parser.add_argument('--person', type=str, default='')
parser.add_argument('--non_chat', '-nc', action='store_true', default=False)
args = parser.parse_args()


filename = f"../review_data/goodreads/titles_1000-2500.npy"

titles = np.load(filename)

def template(title):
    return f'Write a comprehensive review of the book titled {title}: '

def template2(title):
    return f'Write a personalized review of the book titled {title}: '

def template3(title):
    return f"Write a book review for the book titled {title}, from the viewpoint of different personas, such as 'aspiring writer', 'history enthusiast', 'teenage sci-fi fan', or 'career-focused parent', etc. Be creative!"    

def template4(title, person):
    return f'Write a book review for the book titled {title} as if you are {person}: '

def template5(title):
    return f"Below is a personalized review of the book titled {title}: "

def template6(title, person):
    return f"I am {person} and I'm writing a review for the book titled {title}. Below is my review: "
    
if args.non_chat:
    if args.person:
        texts = [template6(title, args.person) for title in titles]
    else:
        texts = [template5(title) for title in titles]
else:
    if args.person:
        texts = [template4(title, args.person) for title in titles]
    else:
        texts = [template2(title) for title in titles]

print(f'total # samples = {len(texts)}')

batch_size = 8

model_path = args.model_path # '../lm-evaluation-harness/models/model_13b_replace'

tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = "[PAD]"
tokenizer.padding_side = "left"
print('load tokenizer done!')

if 'falcon' in model_path:
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
elif 'zephyr' in model_path:
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda()
else:
    model = transformers.AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, torch_dtype="auto").cuda()
    if '7b' in model_path:
        model = model.bfloat16().cuda()
print('load model done!')

all_generated_texts = []
n_batch = (len(texts) - 1) // batch_size + 1
cnt = 0
while osp.exists(f'{args.output_path}_{cnt}.json'):
    cnt += 1
for i in tqdm(range(cnt*20, n_batch)):
    cur_texts = texts[i*batch_size : min((i+1)*batch_size, len(texts))]
    inputs = tokenizer(cur_texts, return_tensors="pt", padding=True, truncation=True, max_length=100)
    inputs.to('cuda:0')
    
    with torch.no_grad():
        if np.isclose(args.temperature, 0):
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens, # 100,
                do_sample=False,
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens, # 100,
                do_sample=True,
                top_p=args.top_p,
                top_k=args.top_k,
                temperature=args.temperature,
            )
    generated_texts = tokenizer.batch_decode(outputs[:,inputs['input_ids'].shape[-1]:], skip_special_tokens=True)
    all_generated_texts += generated_texts
    
    if (i+1) % 20 == 0:
        d = {"completions": all_generated_texts}
        # with open("completions_replace.json", "w") as f:
        with open(f'{args.output_path}_{cnt}.json', "w") as f:
            json.dump(d, f)
            
        all_generated_texts = []
        cnt += 1
        # break
        
if all_generated_texts:
    d = {"completions": all_generated_texts}
    # with open("completions_replace.json", "w") as f:
    with open(f'{args.output_path}_{cnt}.json', "w") as f:
        json.dump(d, f)
        
    all_generated_texts = []