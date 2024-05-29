import argparse
import numpy as np
from itertools import combinations
import json

N = 20

input_filename = './codeforces_A_file_paths.txt'
file_paths = []
with open(input_filename, 'r') as f:
    for line in f:
        file_paths.append(line.strip())
file_paths = file_paths[:100]

def jaccard_index(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 1

def jaccard_similarity_index(list_of_sets):
    if len(list_of_sets) < 2:
        return 0  # Cannot calculate similarity for less than 2 sets
    pairwise_jaccard_indices = [jaccard_index(set1, set2) for set1, set2 in combinations(list_of_sets, 2)]
    average_jaccard_index = sum(pairwise_jaccard_indices) / len(pairwise_jaccard_indices)
    return average_jaccard_index

sim_scores = []

for file_path in file_paths:
    fileid = file_path.split('_')[0]
    pid = file_path.split('_')[-1].split('.')[0]
    
    tags_list = []
    for i in range(N):
        input_file = f'parsed_description_{fileid}_{pid}_solution_{i}.json'
        cur_tags = json.load(open(input_file))['tags'].split(', ')
        tags_list.append(set(list(np.unique(cur_tags))))
    
    similarity = jaccard_similarity_index(tags_list)
    sim_scores.append(similarity)
    print(f"similarity index: {similarity} of {fileid} {pid}")

out_file = f'tags_train_solution_similarity.npy'
print(out_file)
np.save(out_file, np.array(sim_scores))