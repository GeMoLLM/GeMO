import numpy as np
t_1a = np.load('mean_len_goodreads_personalized_llama-2-13b-chat_500.npy')
t_2a = np.load('mean_len_goodreads_personation_llama-2-13b-chat_500.npy')

t_1b = np.load('mean_len_goodreads_personalized_vicuna-13b-chat_500.npy')
t_2b = np.load('mean_len_goodreads_personation_vicuna-13b-chat_500.npy')

t_1c = np.load('mean_len_goodreads_personalized_gpt-4-chat_500.npy')
t_2c = np.load('mean_len_goodreads_personation_gpt-4-chat_500.npy')

for l_1a, l_2a in zip(t_1a, t_2a):
    for x, y in zip(l_1a[:-1], l_2a[:-1]):
        print(f'{x:.2f},{y:.2f}',end=',')
    print(f'{l_1a[-1]:.2f},{l_2a[-1]:.2f}')
print()

for l_1b, l_2b in zip(t_1b, t_2b):
    for x, y in zip(l_1b[:-1], l_2b[:-1]):
        print(f'{x:.2f},{y:.2f}',end=',')
    print(f'{l_1b[-1]:.2f},{l_2b[-1]:.2f}')
print()

for l_1c, l_2c in zip(t_1c, t_2c):
    for x, y in zip(l_1c[:-1], l_2c[:-1]):
        print(f'{x:.2f},{y:.2f}',end=',')
    print(f'{l_1c[-1]:.2f},{l_2c[-1]:.2f}')