import json
import numpy as np

def calculate_entropy(l):
    l = np.array(l) / np.sum(l)
    l = [x for x in l if x > 0]
    return -np.sum(l * np.log(l))

p_solutions = []
p_incorrect_solutions = []    
for i in range(81):
    filename = f'valid_problem_{i}.json'
    data = json.load(open(filename))
    n_solutions_language = data['n_solutions_language']
    n_incorrect_solutions_language = data['n_incorrect_solutions_language']
    
    p_solutions.append(data['n_solution_py3'] / sum(n_solutions_language))
    p_incorrect_solutions.append(data['n_incorrect_solution_py3'] / sum(n_incorrect_solutions_language))
    
np.savez('language_python_portion.npz', p_solutions=p_solutions, p_incorrect_solutions=p_incorrect_solutions)
    