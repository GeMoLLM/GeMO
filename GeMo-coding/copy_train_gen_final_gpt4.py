import os
import os.path as osp
import shutil
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_id', type=str, default='codeonly')
parser.add_argument('--model_fam', type=str, default='gpt4')
parser.add_argument('--idx_fileid', type=str, default='')
parser.add_argument('--input_filename', type=str, default='codeforces_A_file_paths_final.txt')
args = parser.parse_args()

input_filename = f'./{args.input_filename}'
train_idx_filename = f'codeforces_A_gen_{args.model_fam}{args.idx_fileid}_index.npy'

file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
# assert len(file_paths) == 100, f'len(file_paths)={len(file_paths)}'

train_idx = np.load(train_idx_filename)

folder = f'train_gen_final_{args.model_fam}_{args.input_id}'
for file_path, indices in zip(file_paths, train_idx):
    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]

    # Specify the path to the new folder
    new_folder_path = osp.join(folder, f'{fileid}_{pid}')
    
    # Create the new folder if it doesn't exist
    os.makedirs(new_folder_path, exist_ok=True)
    source_paths = [f'{fileid}_solution_{args.input_id}_{i}_{pid}.py' for i in indices]
        
    # Copy files to the new folder
    for source_path in source_paths:
        source_name = os.path.basename(source_path)
        dest_path = os.path.join(new_folder_path, source_name)
        try:
            shutil.copy(source_path, dest_path)
            # print(f"Copied {source_path} to {dest_path}")
        except FileExistsError:
            print(f"File {source_path} already exists at {dest_path}")
        except Exception as e:
            print(f"Error copying {source_path} to {dest_path}: {e}")
