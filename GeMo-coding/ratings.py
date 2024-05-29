import json
import numpy as np

ratings = []
for i in range(81):
    filename = f'valid_problem_info_{i}.json'
    d = json.load(open(filename))
    assert d['source'] == ['CODEFORCES'], f'i={i} source={d["source"]}'
    ratings.append(d['cf_rating'])
    
np.save('ratings.npy', ratings)