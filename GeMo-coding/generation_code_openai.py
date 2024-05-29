import argparse
import json
from tqdm import tqdm
import torch
import numpy as np

from openai import AzureOpenAI
    
client = AzureOpenAI(
    api_key=API_KEY,  
    api_version="2023-12-01-preview",
    azure_endpoint = "https://monoculture.openai.azure.com/"
    )
    
deployment_name='monoculture-gpt-35-turbo-instruct-0914' #This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment. 
    

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='BOLD_completions_debug')
parser.add_argument('--max_new_tokens', type=int, default=2048)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--top_k', type=int, default=50)
parser.add_argument('--person', type=str, default='')
args = parser.parse_args()

# Send a completion call to generate an answer

filename = f'/scratch/fanw6/code_contests/problem.json'

data = json.load(open(filename))
print(data['description'])

print('------------------------------------------------------------------------------')
prompt = "Please explain to me how you understand the following programming question: \n\n" + data['description']

response = client.completions.create(
    model=deployment_name, 
    prompt=prompt, 
    max_tokens=args.max_new_tokens, 
    temperature=args.temperature, 
    top_p=args.top_p
)
solution = response.choices[0].text

print(response)

print(solution)
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')

# def template(title):
#     return f'Write a comprehensive review of the book titled {title}: '

# def template2(title):
#     return f'Write a personalized review of the book titled {title}: '

# def template3(title):
#     return f"Write a book review for the book titled {title}, from the viewpoint of different personas, such as 'aspiring writer', 'history enthusiast', 'teenage sci-fi fan', or 'career-focused parent', etc. Be creative!"    

# def template4(title, person):
#     return f'Write a book review for the book titled {title} as if you are {person}: '

# if args.person:
#     texts = [template4(title, args.person) for title in titles]
# else:
#     texts = [template2(title) for title in titles]

# print(f'total # samples = {len(texts)}')

# all_generated_texts = []
# cnt = 0
# for prompt in tqdm(texts):
#     response = client.completions.create(
#         model=deployment_name, 
#         prompt=prompt, 
#         max_tokens=args.max_new_tokens, 
#         temperature=args.temperature, 
#         top_p=args.top_p
#     )
#     all_generated_texts.append(response.choices[0].text)
    
#     if len(all_generated_texts) == 160:
#         d = {"completions": all_generated_texts}
#         # with open("completions_replace.json", "w") as f:
#         with open(f'{args.output_path}_{cnt}.json', "w") as f:
#             json.dump(d, f)
            
#         all_generated_texts = []
#         cnt += 1
#         # break
        
# if all_generated_texts:
#     d = {"completions": all_generated_texts}
#     # with open("completions_replace.json", "w") as f:
#     with open(f'{args.output_path}_{cnt}.json', "w") as f:
#         json.dump(d, f)
        
#     all_generated_texts = []