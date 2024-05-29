import argparse
import numpy as np
from itertools import combinations

parser = argparse.ArgumentParser()
parser.add_argument('--input_id', type=str, default='codeonly')
parser.add_argument('--n_gen', type=int, default=5)
args = parser.parse_args()

indexs = np.load('./indexs.npy')
indices = np.where(indexs == 'A')[0]

def jaccard_index(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 1

def jaccard_diversity_index(list_of_sets):
    if len(list_of_sets) < 2:
        return 0  # Cannot calculate diversity for less than 2 sets
    pairwise_jaccard_indices = [jaccard_index(set1, set2) for set1, set2 in combinations(list_of_sets, 2)]
    average_jaccard_index = sum(pairwise_jaccard_indices) / len(pairwise_jaccard_indices)
    return 1 - average_jaccard_index

div_scores = []
for pid in indices:
    tags_list = []
    for gen_id in range(1, args.n_gen+1):
        input_file = f'tags_valid_solution_{args.input_id}_{gen_id}_{pid}.txt'
        lines = open(input_file).readlines()
        assert len(lines) == 2
        
        filtered_lines = []
        for line in lines:
            if line.strip() != '':
                filtered_lines.append(line)
        assert len(filtered_lines) == 1
        
        if 'tags: ' in filtered_lines[0]:
            print(filtered_lines[0])
            filtered_lines[0] = filtered_lines[0].replace('tags: ', '')
            
        assert 'tags' not in filtered_lines[0]
            
        cur_tags = filtered_lines[0].split(', ')
        tags_list.append(set(list(np.unique(cur_tags))))
    
    diversity = jaccard_diversity_index(tags_list)
    div_scores.append(diversity)
    print(f"Diversity index: {diversity} of {pid}")

out_file = f'tags_valid_solution_{args.input_id}_diversity.npy'
print(out_file)
np.save(out_file, np.array(div_scores))