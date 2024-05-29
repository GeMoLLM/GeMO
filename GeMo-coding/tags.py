import json
import numpy as np

tags = []
for i in range(81):
    filename = f'valid_problem_info_{i}.json'
    cf_tags = json.load(open(filename))['cf_tags']
    cf_tags = cf_tags.replace('\'', '\"')
    tags.append(json.loads(cf_tags))

print(tags)

tags = [x for l in tags for x in l]
tags = np.unique(tags)
print(tags)
np.save('tags.npy', tags)