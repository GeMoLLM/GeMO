import numpy as np
folder = 'judge_output'
arr = np.zeros((81, 3), dtype=int)
for i in range(81):
    filename = f'{folder}/valid_solution_{i}.py'
    lines = open(filename).readlines()
    compile = int(lines[0].strip())
    n_pass = int(lines[1].strip())
    runtime = int(lines[2].strip())
    arr[i] = (compile, n_pass, runtime)
    print(compile, n_pass, runtime)
    
np.save('judge_output_gpt4_gen_1.npy', arr)