import numpy as np

indices = np.load('sel_indices_claude.npy')

new_lines = []
with open('codeforces_A_file_paths_final.txt') as f:
    lines = f.readlines()
    assert len(lines) == 100
    for idx in indices:
        new_lines.append(lines[idx])
        
with open('codeforces_A_file_paths_claude_sel.txt', 'w') as f:
    f.writelines(new_lines)