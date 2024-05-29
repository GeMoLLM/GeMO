import json
import numpy as np

ratings = []
for i in range(81):
    filename = f'valid_problem_info_{i}.json'
    d = json.load(open(filename))
    ratings.append(d['cf_index'][0])
    
np.save('indexs.npy', ratings)