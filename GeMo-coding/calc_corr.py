import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau

def obtain_corr(l1, l2):
    return spearmanr(l1, l2).statistic, pearsonr(l1, l2)[0], kendalltau(l1, l2).correlation

data = np.load('judge_output.npz')
data_src = data['cnt_cor'].mean(axis=1)
data_gpt4_1 = np.load('judge_output_gpt4_gen_1.npy')[:,1]
data_gpt4_2 = np.load('judge_output_gpt4_gen_2.npy')[:,1]
data_gpt4_3 = np.load('judge_output_gpt4_gen_3.npy')[:,1]
data_gpt4_4 = np.load('judge_output_gpt4_gen_4.npy')[:,1]
ratings = np.load('ratings.npy')

print(data_gpt4_1)
print(ratings)

result = np.zeros((5, 3))
result[0] = obtain_corr(data_src, ratings)
result[1] = obtain_corr(data_gpt4_1, ratings)
result[2] = obtain_corr(data_gpt4_2, ratings)
result[3] = obtain_corr(data_gpt4_3, ratings)
result[4] = obtain_corr(data_gpt4_4, ratings)

np.set_printoptions(precision=3)

print(result)

s = [[str(e) for e in row] for row in result]
lens = [max(map(len, col)) for col in zip(*s)]
fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
table = [fmt.format(*row) for row in s]

print('\n'.join(table))

data_gpt4_i = [data_gpt4_1, data_gpt4_2, data_gpt4_3, data_gpt4_4]

mut_corr = np.zeros((4, 4, 3))
for i in range(4):
    for j in range(i+1, 4):
        mut_corr[i][j] = obtain_corr(data_gpt4_i[i], data_gpt4_i[j])

print(mut_corr[:,:,0])
print(mut_corr[:,:,1])
print(mut_corr[:,:,2])