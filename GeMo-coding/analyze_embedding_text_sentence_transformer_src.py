import argparse
import json
import os
import os.path as osp
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import numpy as np

input_filename = './codeforces_A_file_paths_final.txt'
train_idx_filename = 'codeforces_A_train_index.npy'

file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
assert len(file_paths) == 100, f'len(file_paths)={len(file_paths)}'

train_idx = np.load(train_idx_filename)

parser = argparse.ArgumentParser()
parser.add_argument('--field', type=str, default='description')
parser.add_argument('--model', type=str, default='gpt-3.5-instruct')
args = parser.parse_args()

model = SentenceTransformer('all-MiniLM-L6-v2').cuda()

def calculate_intra_pairwise_sim(desc):
    embeddings = model.encode(desc, convert_to_tensor=True)
    cosine_scores = util.cos_sim(embeddings, embeddings)
    n = len(cosine_scores)
    return (cosine_scores.sum().item() - n) / (n * (n - 1))

sim_scores = []
for file_path, indices in tqdm(zip(file_paths, train_idx)):
    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]
    
    text_list = []
    for i in indices:
        input_file = f'parsed_description_{args.model}_{fileid}_{pid}_solution_{i}.json'
        cur_text = json.load(open(input_file))[args.field]
        text_list.append(cur_text)
        
    sim_scores.append(calculate_intra_pairwise_sim(text_list))

out_file = f'pairwise_sim_scores_{args.field}_{args.model}_src_final.npy'
print(out_file)
np.save(out_file, np.array(sim_scores))
