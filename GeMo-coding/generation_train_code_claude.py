import argparse
import json
from tqdm import tqdm
import torch
import numpy as np
import os.path as osp
import openai
import time
import anthropic

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='BOLD_completions_debug')
parser.add_argument('--max_new_tokens', type=int, default=2048)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--top_k', type=int, default=50)
parser.add_argument('--person', type=str, default='')
parser.add_argument('--output_id', type=str, default='persona_codeonly')
parser.add_argument('--begin_id', type=int, default=0)
parser.add_argument('--end_id', type=int, default=100)
parser.add_argument('--input_filename', default='', type=str)
parser.add_argument('--key_id', type=int, default=0)
args = parser.parse_args()

if args.key_id == 0:
    api_key=API_KEY
elif args.key_id == 1:
    api_key = API_KEY
elif args.key_id == 2:
    api_key = API_KEY
else:
    raise NotImplementedError('key_id not implemented')

client = anthropic.Anthropic(
    # defaults to os.environ.get("ANTHROPIC_API_KEY")
    api_key=api_key,
)

if args.input_filename:
    input_filename = args.input_filename
else:
    input_filename = '/scratch/fanw6/code_contests/codeforces_A_file_paths_final.txt'

file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
assert len(file_paths) <= 100

folder = '/home/fanw6/main/code_contests/'

# Send a completion call to generate an answer

def generate_response(prompt):
    try:
        response = client.messages.create(
            model="claude-3-sonnet-20240229",
            system="Assistant is a code language model capable of solving complicated competitive programming problems on platforms such as Codeforces.",
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        print("Rate limit exceeded, retrying in 10 seconds...")
        time.sleep(60)
        return generate_response(prompt)

i = 0
while i < len(file_paths):
    fileid = file_paths[i].split('_')[0]
    pid = file_paths[i].split('_')[-1].split('.')[0]
    out_path = f'/scratch/fanw6/code_contests/{fileid}_solution_{args.output_id}_{pid}.txt'
    if osp.exists(out_path):
        print(f'{out_path} exists, skipping...{i}')
        i += 1
    else:
        break

for file_path in tqdm(file_paths[i:]):
    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]

    file_path = osp.join(folder, file_path)
    data = json.load(open(file_path))
    
    # prompt = 'Please read the below problem description and generate a python code to solve the problem:\n\n' + data['description']
    prompt = 'Please read the below problem description and generate a python code to solve the problem:\n\n' + data['description'] + '\n\nPlease only generate code and nothing else.'
    
    solution = generate_response(prompt)
    # prompt = "Imagine you are a grandmaster in solving competitive programming problems. Your skills in algorithms, data structures, and problem-solving are unparalleled. You have a deep understanding of various programming paradigms and can easily navigate through complex problems with efficiency and elegance.\n\nPlease read the below problem description and generate a python code to solve the problem:\n\n" + data['description'] + '\n\nPlease only generate code and nothing else.'
    
    # prompt = "Imagine you are a grandmaster in solving competitive programming problems. Your skills in algorithms, data structures, and problem-solving are unparalleled. You have a deep understanding of various programming paradigms and can easily navigate through complex problems with efficiency and elegance.\n\nPlease read the below problem description and generate a python code to solve the problem:\n\n" + data['description'] + '\n\nPlease think through the problem step by step, and then provide your solution. Then, test your code against the provided test cases in the problem. If your code fails to pass all the tests, please revise your code and try again until your code passes all the tests.'

    with open(f'/scratch/fanw6/code_contests/{fileid}_solution_{args.output_id}_{pid}.txt', 'w') as f:
        f.write(solution)
