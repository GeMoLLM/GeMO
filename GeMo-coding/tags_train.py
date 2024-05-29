import json
import numpy as np
import os.path as osp

input_filename = './codeforces_A_file_paths.txt'
file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())

folder = './'

tags = []
for file_path in file_paths:
    file_path = osp.join(folder, file_path)
    cf_tags = json.load(open(file_path))['cf_tags']
    cf_tags = cf_tags.replace('\'', '\"')
    tags.append(json.loads(cf_tags))

print(tags)

tags = [x for l in tags for x in l]
tags = np.unique(tags)
print(tags)
np.save('tags_train.npy', tags)