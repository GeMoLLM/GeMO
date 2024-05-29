import argparse
from openai import AzureOpenAI
import numpy as np
from tqdm import tqdm
import os.path as osp

folder = '/scratch/fanw6/code_contests/'
input_filename = osp.join(folder, 'codeforces_A_file_paths_final.txt')
train_idx_filename = osp.join(folder, 'codeforces_A_train_index.npy')

file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
assert len(file_paths) == 100, f'len(file_paths)={len(file_paths)}'

train_idx = np.load(train_idx_filename)

client = AzureOpenAI(
    api_key=API_KEY,  
    api_version="2023-12-01-preview",
    azure_endpoint = "https://monoculture.openai.azure.com/"
)
deployment_name = 'monoculture-gpt-35-turbo-instruct-0914' #This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment. 

instruction = "Please provide a description to the following code in natural language. Explain the functionality, algorithm, data structure, time complexity, space complexity of the code.\n"\
"Finally, assign a few tags to the code. Here is a list of tags you can choose from:\n\"binary search, math, special, trees, dp, greedy, games, dfs and similar, expression parsing, number theory, chinese remainder theorem, geometry, bitmasks, sortings, graph matchings, matrices, meet-in-the-middle, graphs, combinatorics, probabilities, constructive algorithms, schedules, two pointers, brute force, dsu, shortest paths, hashing, interactive, data structures, strings, ternary search, fft, flows, implementation\"\n"\
"Answer each in a line in the example format of: 'Description: description\\nFunctionality: functionality\\n'\n"

print(instruction)
for file_path, indices in tqdm(zip(file_paths, train_idx)):
    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]

    for i in tqdm(indices):

        lines = open(osp.join(folder, f"{fileid}_{pid}_solutions_{i}.txt")).readlines()
        code = ''.join(lines)

        prompt = instruction + code
            
        response = client.completions.create(model=deployment_name, prompt=prompt, max_tokens=500, temperature=0)

        outfile_name = osp.join(folder, f'description_gpt-3.5-instruct_{fileid}_{pid}_solution_{i}.txt')
        with open(outfile_name, 'w') as f:
            f.write(response.choices[0].text)