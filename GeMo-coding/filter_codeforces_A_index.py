import numpy as np
import glob
import json

pattern = "train-*_problem_info*"
file_paths = glob.glob(pattern)
print(len(file_paths))

valid_file_paths = []
for file_path in file_paths:
    d = json.load(open(file_path))
    if d['source'] != ["CODEFORCES"]:
        continue
    
    if d['cf_index'].startswith('A'):
        valid_file_paths.append(file_path)
        
print(len(valid_file_paths), len(file_paths))
with open('codeforces_A_file_paths.txt', 'w') as f:
    for file_path in valid_file_paths:
        f.write(file_path + '\n')