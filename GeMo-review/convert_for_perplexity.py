import argparse
import json
import os
import os.path as osp
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_prefix', type=str, default='goodreads_completions_llama-2-13b-chat_500')
parser.add_argument('--output_dir', type=str, default='')
parser.add_argument('--personation', action='store_true', default=False)
args = parser.parse_args()

gen0 = json.load(open(f'{args.input_prefix}-0_merged.json'))
gen1 = json.load(open(f'{args.input_prefix}-1_merged.json'))
gen2 = json.load(open(f'{args.input_prefix}-2_merged.json'))
gen3 = json.load(open(f'{args.input_prefix}-3_merged.json'))
gen4 = json.load(open(f'{args.input_prefix}-4_merged.json'))
gen_all = []
for a,b,c,d,e in zip(gen0,gen1,gen2,gen3,gen4):
    gen_all += [a,b,c,d,e]

def template2(title):
    return f'Write a personalized review of the book titled {title}: '

def template4(title, person):
    return f'Write a book review for the book titled {title} as if you are {person}: '

filename = f"../review_data/goodreads/titles_1000-2500.npy"

titles = np.load(filename)

if args.personation:
    persons = ["Trevor Noah", "Janelle Monáe", "Yuval Noah Harari", "Serena Williams", "Reshma Saujani"]
    texts = [template4(title, persons[i]) for title in titles for i in range(5)]
else:
    texts = [template2(title) for title in titles for _ in range(5)]

print(len(gen_all))

import json

print(texts[:20])

json_data = {
    "type": "text2text",
    "instances": [{"input": input, "output": output} for input, output in zip(texts, gen_all)]
}

output_dir = osp.join('data_perplexity', args.output_dir)
if not osp.exists(output_dir):
    os.makedirs(output_dir)
json.dump(json_data, open(osp.join(output_dir, 'data.json'), 'w'), indent=2)