import numpy as np
t_1a = np.load('mean_len_goodreads_personalized_llama-2-13b-nonchat_500.npy')
t_2a = np.load('mean_len_goodreads_personation_llama-2-13b-nonchat_500.npy')

for l_1a, l_2a in zip(t_1a, t_2a):
    for x, y in zip(l_1a[:-1], l_2a[:-1]):
        print(f'{x:.2f},{y:.2f}',end=',')
    print(f'{l_1a[-1]:.2f},{l_2a[-1]:.2f}')
print()
