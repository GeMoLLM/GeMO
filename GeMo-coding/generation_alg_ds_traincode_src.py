import argparse
from openai import AzureOpenAI
import numpy as np
from tqdm import tqdm
import os.path as osp

input_filename = '/scratch/fanw6/code_contests/codeforces_A_file_paths.txt'
file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
file_paths = file_paths[:100]

folder = '/home/fanw6/main/code_contests/'

client = AzureOpenAI(
    api_key=API_KEY,  
    api_version="2023-12-01-preview",
    azure_endpoint = "https://monoculture.openai.azure.com/"
)
deployment_name = 'monoculture-gpt-35-turbo-instruct-0914' #This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment. 

instruction = "Please read the following code and infer the algorithms and data structures used in it.\n"\
"For algorithms, select (a few) from the following list: \n"\
"\"Sorting Algorithms, Searching Algorithms, String Algorithms, Divide and Conquer Algorithms, Greedy Algorithms, Dynamic Programming, Recursion, Bit Manipulation, Backtracking, Graph Algorithms, Others\"\n"\
"For data structures, select (a few) from the following list: \n"\
"\"Arrays, Linked Lists, Stacks, Queues, Trees, Heaps, Hash Tables, Sets, Maps, Priority Queues, Others\"\n"\
"Answer each in a line following the format of: 'Algorithms: candidate 1, candidiate 2, ..\\nData structures: candidate 1, candidiate 2, ..\\n'\n"

for file_path in tqdm(file_paths):
    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]

    for i in tqdm(range(20)):

        lines = open(f"/scratch/fanw6/code_contests/{fileid}_{pid}_solutions_{i}.txt").readlines()
        code = ''.join(lines)

        prompt = instruction + code
            
        response = client.completions.create(model=deployment_name, prompt=prompt, max_tokens=500, temperature=0)

        outfile_name = f'/scratch/fanw6/code_contests/alg_ds_{fileid}_{pid}_solution_{i}.txt'
        with open(outfile_name, 'w') as f:
            f.write(response.choices[0].text)