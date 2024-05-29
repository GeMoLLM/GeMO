import argparse
import json        
import numpy as np

n_cor = np.zeros((81), dtype=int)
n_inc = np.zeros((81), dtype=int)
n_py_cor = np.zeros((81), dtype=int)
n_py_inc = np.zeros((81), dtype=int)
for i in range(81):
    data = json.load(open(f'valid_problem_{i}.json'))
    n_cor[i] = data['n_solution']
    n_inc[i] = data['n_incorrect_solution']
    n_py_cor[i] = data['n_solution_py3']
    n_py_inc[i] = data['n_incorrect_solution_py3']
    
np.savez('acc.npz', n_cor=n_cor, n_inc=n_inc, n_py_cor=n_py_cor, n_py_inc=n_py_inc)