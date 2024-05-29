import json
import numpy as np

def calculate_entropy(l):
    l = np.array(l) / np.sum(l)
    l = [x for x in l if x > 0]
    return -np.sum(l * np.log(l))

entropy_solutions = []
entropy_incorrect_solutions = []    
for i in range(81):
    filename = f'valid_problem_{i}.json'
    data = json.load(open(filename))
    n_solutions_language = data['n_solutions_language']
    n_incorrect_solutions_language = data['n_incorrect_solutions_language']
    
    entropy_solutions.append(calculate_entropy(n_solutions_language))
    entropy_incorrect_solutions.append(calculate_entropy(n_incorrect_solutions_language))
    
np.savez('entropy.npz', entropy_solutions=entropy_solutions, entropy_incorrect_solutions=entropy_incorrect_solutions)
    