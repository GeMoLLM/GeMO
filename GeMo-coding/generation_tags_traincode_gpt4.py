import argparse
from openai import AzureOpenAI
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_id', type=str, default='codeonly_1')
args = parser.parse_args()

indexs = np.load('/scratch/fanw6/code_contests/indexs.npy')
indices = np.where(indexs == 'A')[0]

instruction = "Please read the code below and assign a few tags to it. Here is a list of tags you can choose from:\nbinary search, math, special, trees, dp, greedy, games, dfs and similar, expression parsing, number theory, chinese remainder theorem, geometry, bitmasks, sortings, graph matchings, matrices, meet-in-the-middle, graphs, combinatorics, probabilities, constructive algorithms, schedules, two pointers, brute force, dsu, shortest paths, hashing, interactive, data structures, strings, ternary search, fft, flows, implementation.\nOutput only the tags and nothing else.\n"

for i in tqdm(indices):
    lines = open(f"/scratch/fanw6/code_contests/valid_solution_{args.input_id}_{i}.py").readlines()
    code = ''.join(lines)

    prompt = instruction + code
        
    client = AzureOpenAI(
        api_key=API_KEY,  
        api_version="2023-12-01-preview",
        azure_endpoint = "https://monoculture.openai.azure.com/"
    )
        
    deployment_name = 'monoculture-gpt-35-turbo-instruct-0914' #This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment. 
        
    response = client.completions.create(model=deployment_name, prompt=prompt, max_tokens=500, logprobs=1, temperature=0)

    outfile_name = f'/scratch/fanw6/code_contests/tags_valid_solution_{args.input_id}_{i}.txt'
    with open(outfile_name, 'w') as f:
        f.write(response.choices[0].text)