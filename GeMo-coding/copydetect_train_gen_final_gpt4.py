import argparse
from copydetect import CopyDetector
import os
import os.path as osp
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_id', type=str, default='codeonly')
parser.add_argument('--model', type=str, default='gpt4')
parser.add_argument('--input_filename', type=str, default='codeforces_A_file_paths_final.txt')
args = parser.parse_args()

input_filename = f'./{args.input_filename}'
file_paths = []
report_folder = f'copydetect_reports_gen_final_{args.model}_{args.input_id}'
scores_folder = f'copydetect_scores_gen_final_{args.model}_{args.input_id}'

os.makedirs(report_folder, exist_ok=True)
os.makedirs(scores_folder, exist_ok=True)

with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
# assert len(file_paths) == 100, f'len(file_paths)={len(file_paths)}'

for file_path in tqdm(file_paths):
    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]

    detector = CopyDetector(
        test_dirs=[f"train_gen_final_{args.model}_{args.input_id}/{fileid}_{pid}"], out_file=osp.join(report_folder, f'report_{fileid}_{pid}.html'))
    detector.run()
    sim_mat = detector.similarity_matrix[:,:,0]
    scores = sim_mat[sim_mat!=-1]
    
    np.save(osp.join(scores_folder, f'scores_{fileid}_{pid}.npy'), scores)
    
    detector.generate_html_report()