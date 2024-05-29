import numpy as np
index = np.load('indexs.npy')
uni_index = np.unique(index)

for idx in uni_index:
    print(f'Index: {idx}')
    print(f'Number of problems: {np.where(index == idx)[0]}, {np.sum(index == idx)}')

data = np.load('judge_output.npz')
data_src = data['cnt_cor'].mean(axis=1)
data_gpt4_1 = np.load('judge_output_gpt4_gen_1.npy')[:,1]
data_gpt4_2 = np.load('judge_output_gpt4_gen_2.npy')[:,1]
data_gpt4_3 = np.load('judge_output_gpt4_gen_3.npy')[:,1]
data_gpt4_4 = np.load('judge_output_gpt4_gen_4.npy')[:,1]
ratings = np.load('ratings.npy')

