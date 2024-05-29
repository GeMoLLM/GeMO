import argparse
import openai
import json
from tqdm import tqdm
import time
import numpy as np
import os.path as osp

from openai import AzureOpenAI
    
client = AzureOpenAI(
    api_key=API_KEY,  
    api_version="2023-12-01-preview",
    azure_endpoint = "https://monoculture.openai.azure.com/"
    )
    
deployment_name = 'monoculture-gpt-4-0125'

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='BOLD_completions_debug')
parser.add_argument('--max_new_tokens', type=int, default=500)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--person', type=str, default='')
args = parser.parse_args()

# Send a completion call to generate an answer

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

if args.person:
    texts = [template4(title, args.person) for title in titles]
else:
    texts = [template2(title) for title in titles]

print(f'total # samples = {len(texts)}')


def generate_response(prompt):
    try:
        response = client.chat.completions.create(
            model=deployment_name, 
            messages=[
                {"role": "system", "content": "Assistant is a language model capable of generating human-like book reviews."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=args.max_new_tokens, 
            temperature=args.temperature, 
            top_p=args.top_p
        )
        return response.choices[0].message.content
    except openai.RateLimitError as e:
        print("Rate limit exceeded, retrying in 10 seconds...")
        time.sleep(10)
        return generate_response(prompt)

i = 0
while osp.exists(f'{args.output_path}_{i}.json'):
    print(f'File exists, skipping...{i}')
    i += 1

all_generated_texts = []
cnt = i*10
for prompt in tqdm(texts[cnt:]):
    all_generated_texts.append(generate_response(prompt))

    if len(all_generated_texts) == 10:
        d = {"completions": all_generated_texts}
        # with open("completions_replace.json", "w") as f:
        with open(f'{args.output_path}_{cnt}.json', "w") as f:
            json.dump(d, f)
            
        all_generated_texts = []
        cnt += 1

if all_generated_texts:
    d = {"completions": all_generated_texts}
    # with open("completions_replace.json", "w") as f:
    with open(f'{args.output_path}_{cnt}.json', "w") as f:
        json.dump(d, f)
        
    all_generated_texts = []