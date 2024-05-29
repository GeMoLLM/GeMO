import numpy as np

indices = np.load('sel_indices_claude.npy')

new_lines = []
with open('run_autojudge_all_final.sh') as f:
    lines = f.readlines()
    assert len(lines) == 102
    new_lines += lines[:2]
    for idx in indices:
        new_lines.append(lines[idx+2])
        
with open('run_autojudge_all_claude_sel.sh', 'w') as f:
    f.writelines(new_lines)