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
    
deployment_name = 'monoculture-gpt-4-0125'
    

parser = argparse.ArgumentParser()
parser.add_argument('--output_path', type=str, default='BOLD_completions_debug')
parser.add_argument('--max_new_tokens', type=int, default=2048)
parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--top_p', type=float, default=0.9)
parser.add_argument('--top_k', type=int, default=50)
parser.add_argument('--person', type=str, default='')
args = parser.parse_args()

# Send a completion call to generate an answer

for i in range(81):
    filename = f'/scratch/fanw6/code_contests/valid_problem_{i}.json'

    data = json.load(open(filename))
    print(data['description'])

    print('------------------------------------------------------------------------------')
    # prompt = 'Please read the below problem description and generate a python code to solve the problem:\n\n' + data['description']
    # prompt = 'Please read the below problem description and generate a python code to solve the problem:\n\n' + data['description'] + '\n\nPlease only generate code and nothing else.'
    
    # prompt = "Imagine you are a grandmaster in solving competitive programming problems. Your skills in algorithms, data structures, and problem-solving are unparalleled. You have a deep understanding of various programming paradigms and can easily navigate through complex problems with efficiency and elegance.\n\nPlease read the below problem description and generate a python code to solve the problem:\n\n" + data['description'] + '\n\nPlease only generate code and nothing else.'
    
    prompt = "Imagine you are a grandmaster in solving competitive programming problems. Your skills in algorithms, data structures, and problem-solving are unparalleled. You have a deep understanding of various programming paradigms and can easily navigate through complex problems with efficiency and elegance.\n\nPlease read the below problem description and generate a python code to solve the problem:\n\n" + data['description'] + '\n\nPlease think through the problem step by step, and then provide your solution. Then, test your code against the provided test cases in the problem. If your code fails to pass all the tests, please revise your code and try again until your code passes all the tests.'

    response = client.chat.completions.create(
        model=deployment_name, # model = "deployment_name".
        messages=[
            {"role": "system", "content": "Assistant is a code language model capable of solving complicated competitive programming problems on platforms such as Codeforces."},
            {"role": "user", "content": prompt}
        ],
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        top_p=args.top_p,
    )

    solution = response.choices[0].message.content

    # print(response)

    print(solution)
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    
    with open(f'/scratch/fanw6/code_contests/valid_solution_{args.output_id}_{i}.txt', 'w') as f:
        f.write(solution)

