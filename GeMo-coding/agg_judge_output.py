import os.path as osp
import numpy as np

folder = 'judge_output/'

cnt_cor = np.zeros((81, 5))
dur_cor = np.zeros((81, 5))
cnt_inc = np.zeros((81, 5))
dur_inc = np.zeros((81, 5))

for i in range(81):
    for j in range(5):
        filename = f'valid_problem_{i}_correct_{j}.txt'
        lines = open(osp.join(folder, filename)).readlines()
        cnt = int(lines[0].strip())
        dur = int(lines[1].strip())
        cnt_cor[i, j] = cnt
        dur_cor[i, j] = dur
        
    for j in range(5):
        filename = f'valid_problem_{i}_incorrect_{j}.txt'
        lines = open(osp.join(folder, filename)).readlines()
        cnt = int(lines[0].strip())
        dur = int(lines[1].strip())
        cnt_inc[i, j] = cnt
        dur_inc[i, j] = dur

np.savez('judge_output.npz', cnt_cor=cnt_cor, dur_cor=dur_cor, cnt_inc=cnt_inc, dur_inc=dur_inc)