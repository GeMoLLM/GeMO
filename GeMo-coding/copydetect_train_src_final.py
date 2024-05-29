from copydetect import CopyDetector
import os
import os.path as osp
from tqdm import tqdm
import numpy as np

input_filename = './codeforces_A_file_paths_final.txt'
file_paths = []
report_folder = 'copydetect_reports_src_final'
scores_folder = 'copydetect_scores_src_final'

os.makedirs(report_folder, exist_ok=True)
os.makedirs(scores_folder, exist_ok=True)

with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
assert len(file_paths) == 100, f'len(file_paths)={len(file_paths)}'

for file_path in tqdm(file_paths):
    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]

    detector = CopyDetector(
        test_dirs=[f"train_solutions_final/{fileid}_{pid}"], out_file=osp.join(report_folder, f'report_{fileid}_{pid}.html'))
    detector.run()
    sim_mat = detector.similarity_matrix[:,:,0]
    scores = sim_mat[sim_mat!=-1]
    
    np.save(osp.join(scores_folder, f'scores_{fileid}_{pid}.npy'), scores)
    
    detector.generate_html_report()