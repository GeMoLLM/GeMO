import argparse
import json
import os
import os.path as osp
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--field', type=str, default='description')
parser.add_argument('--model', type=str, default='gpt-3.5-instruct')
parser.add_argument('--input_id', type=str, default='codeonly')
parser.add_argument('--model_fam', type=str, default='gpt4')
parser.add_argument('--idx_fileid', type=str, default='')
parser.add_argument('--input_filename', type=str, default='codeforces_A_file_paths_final.txt')
parser.add_argument('--device', '-d', type=int, default=0)
args = parser.parse_args()

os.environ['TRANSFORMERS_CACHE'] = '../cache/huggingface/'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

input_filename = f'./{args.input_filename}'
train_idx_filename = f'codeforces_A_gen_{args.model_fam}{args.idx_fileid}_index.npy'

file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
# assert len(file_paths) == 100, f'len(file_paths)={len(file_paths)}'

train_idx = np.load(train_idx_filename)

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
        input_file = f'parsed_description_{args.model}_{fileid}_solution_{args.input_id}_{i}_{pid}.json'
        cur_text = json.load(open(input_file))[args.field]
        text_list.append(cur_text)
        
    sim_scores.append(calculate_intra_pairwise_sim(text_list))

out_file = f'pairwise_sim_scores_{args.field}_{args.model}_gen_{args.input_id}_final.npy'
print(out_file)
np.save(out_file, np.array(sim_scores))
