import argparse
from openai import AzureOpenAI
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--input_id', type=str, default='codeonly_1')
args = parser.parse_args()

indexs = np.load('/scratch/fanw6/code_contests/indexs.npy')
indices = np.where(indexs == 'A')[0]

instruction = "Please provide a description to the following code in natural language. Explain the functionality, algorithm, data structure, time complexity, space complexity of the code.\nAnswer each in a line in the example format of: 'Description: description\\nFunctionality: functionality\\n'"

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

    outfile_name = f'/scratch/fanw6/code_contests/description_valid_solution_{args.input_id}_{i}.txt'
    with open(outfile_name, 'w') as f:
        f.write(response.choices[0].text)