# generation
bash run_batch_generation_claude.sh claude_codeonly_temp-1.0_p-0.9 1.0 0.9 0 39
bash run_batch_generation_claude.sh claude_codeonly_temp-1.0_p-0.9 1.0 0.9 11 39
bash run_batch_generation_claude.sh claude_codeonly_temp-1.0_p-0.9 1.0 0.9 21 39

bash run_batch_generation_claude.sh claude_codeonly_temp-1.0_p-0.9 1.0 0.9 34 59 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_sel.txt"

bash run_batch_generation_claude.sh claude_codeonly_temp-1.0_p-0.9 1.0 0.9 60 79 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_sel.txt"

bash run_batch_generation_claude.sh claude_codeonly_temp-1.0_p-0.9 1.0 0.9 79 99 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_sel.txt"

bash run_batch_generation_claude.sh claude_codeonly_temp-1.0_p-0.9 1.0 0.9 99 119 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_sel.txt"
bash run_batch_generation_claude.sh claude_codeonly_temp-1.0_p-0.9 1.0 0.9 120 139 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_sel.txt"


# generation (temp=0.5)

bash run_batch_generation_claude.sh claude_codeonly_temp-0.5_p-0.9 0.5 0.9 0 39
bash run_batch_generation_claude.sh claude_codeonly_temp-0.5_p-0.9 0.5 0.9 10 49
bash run_batch_generation_claude.sh claude_codeonly_temp-0.5_p-0.9 0.5 0.9 20 49 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_final.txt"
bash run_batch_generation_claude.sh claude_codeonly_temp-0.5_p-0.9 0.5 0.9 50 79 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_final.txt"
bash run_batch_generation_claude.sh claude_codeonly_temp-0.5_p-0.9 0.5 0.9 80 119 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_final.txt"
bash run_batch_generation_claude.sh claude_codeonly_temp-0.5_p-0.9 0.5 0.9 114 150 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_final.txt"
bash run_batch_generation_claude.sh claude_codeonly_temp-0.5_p-0.9 0.5 0.9 151 179 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_sel_2.txt"
bash run_batch_generation_claude.sh claude_codeonly_temp-0.5_p-0.9 0.5 0.9 180 199 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_sel_2.txt"

# generation (temp=0.5, p=1.0)
bash run_batch_generation_claude.sh claude_codeonly_temp-0.5_p-1.0 0.5 1.0 0 99 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_final.txt --key_id 1"
bash run_batch_generation_claude.sh claude_codeonly_temp-0.5_p-1.0 0.5 1.0 50 99 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_final.txt --key_id 0"
bash run_batch_generation_claude.sh claude_codeonly_temp-0.5_p-1.0 0.5 1.0 70 99 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_final.txt --key_id 1"
bash run_batch_generation_claude.sh claude_codeonly_temp-0.5_p-1.0 0.5 1.0 97 119 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_final.txt --key_id 1"
bash run_batch_generation_claude.sh claude_codeonly_temp-0.5_p-1.0 0.5 1.0 120 149 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_final.txt --key_id 1"
bash run_batch_generation_claude.sh claude_codeonly_temp-0.5_p-1.0 0.5 1.0 148 149 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_final.txt --key_id 1"


# generation (temp=1.0, p=1.0)
bash run_batch_generation_claude.sh claude_codeonly_temp-1.0_p-1.0 1.0 1.0 0 30 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_final.txt --key_id 1"
bash run_batch_generation_claude.sh claude_codeonly_temp-1.0_p-1.0 1.0 1.0 31 49 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_final.txt --key_id 2"
bash run_batch_generation_claude.sh claude_codeonly_temp-1.0_p-1.0 1.0 1.0 50 59 "--input_filename /scratch/fanw6/code_contests/codeforces_A_file_paths_claude_final.txt --key_id 1"


# convert
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-1.0_p-0.9 0 19
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-1.0_p-0.9 20 33
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-1.0_p-0.9 34 59 "--input_filename codeforces_A_file_paths_claude_sel.txt"
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-1.0_p-0.9 60 79 "--input_filename codeforces_A_file_paths_claude_sel.txt"
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-1.0_p-0.9 80 98 "--input_filename codeforces_A_file_paths_claude_sel.txt"
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-1.0_p-0.9 99 119 "--input_filename codeforces_A_file_paths_claude_sel.txt"
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-1.0_p-0.9 120 139 "--input_filename codeforces_A_file_paths_claude_sel.txt"

# convert (temp=0.5)
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-0.5_p-0.9 0 14
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-0.5_p-0.9 15 26

bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-0.5_p-0.9 27 49 "--input_filename codeforces_A_file_paths_claude_final.txt"
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-0.5_p-0.9 50 113 "--input_filename codeforces_A_file_paths_claude_final.txt"
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-0.5_p-0.9 114 149 "--input_filename codeforces_A_file_paths_claude_final.txt"
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-0.5_p-0.9 150 179 "--input_filename codeforces_A_file_paths_claude_sel_2.txt"
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-0.5_p-0.9 180 199 "--input_filename codeforces_A_file_paths_claude_sel_2.txt"

# convert (temp=0.5, p=1.0)
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-0.5_p-1.0 0 29 "--input_filename codeforces_A_file_paths_claude_final.txt"
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-0.5_p-1.0 30 49 "--input_filename codeforces_A_file_paths_claude_final.txt"
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-0.5_p-1.0 50 69 "--input_filename codeforces_A_file_paths_claude_final.txt"
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-0.5_p-1.0 70 89 "--input_filename codeforces_A_file_paths_claude_final.txt"
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-0.5_p-1.0 90 119 "--input_filename codeforces_A_file_paths_claude_final.txt"
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-0.5_p-1.0 120 149 "--input_filename codeforces_A_file_paths_claude_final.txt"
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-0.5_p-1.0 150 199 "--input_filename codeforces_A_file_paths_claude_final.txt"

# convert (temp=1.0, p=1.0)
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-1.0_p-1.0 0 109 "--input_filename codeforces_A_file_paths_claude_final.txt"
bash run_batch_process.sh parse_gpt4_train_gen_final.py claude_codeonly_temp-1.0_p-1.0 110 199 "--input_filename codeforces_A_file_paths_claude_final.txt"


# test
time bash run_bash_process_batch.sh run_autojudge_all_final.sh claude_codeonly_temp-1.0_p-0.9 0 19

time bash run_bash_process_batch.sh run_autojudge_all_final.sh claude_codeonly_temp-1.0_p-0.9 15 19

time bash run_bash_process_batch.sh run_autojudge_all_final.sh claude_codeonly_temp-1.0_p-0.9 10 14

time bash run_bash_process_batch.sh run_autojudge_all_final.sh claude_codeonly_temp-1.0_p-0.9 5 9

time bash run_bash_process_batch.sh run_autojudge_all_final.sh claude_codeonly_temp-1.0_p-0.9 20 24
time bash run_bash_process_batch.sh run_autojudge_all_final.sh claude_codeonly_temp-1.0_p-0.9 25 29
time bash run_bash_process_batch.sh run_autojudge_all_final.sh claude_codeonly_temp-1.0_p-0.9 30 33

time bash run_bash_process_batch.sh run_autojudge_all_claude_sel.sh claude_codeonly_temp-1.0_p-0.9 33 39
time bash run_bash_process_batch.sh run_autojudge_all_claude_sel.sh claude_codeonly_temp-1.0_p-0.9 40 49
time bash run_bash_process_batch.sh run_autojudge_all_claude_sel.sh claude_codeonly_temp-1.0_p-0.9 50 59

time bash run_bash_process_batch.sh run_autojudge_all_claude_sel.sh claude_codeonly_temp-1.0_p-0.9 60 69
time bash run_bash_process_batch.sh run_autojudge_all_claude_sel.sh claude_codeonly_temp-1.0_p-0.9 70 79

time bash run_bash_process_batch.sh run_autojudge_all_claude_sel.sh claude_codeonly_temp-1.0_p-0.9 80 89
time bash run_bash_process_batch.sh run_autojudge_all_claude_sel.sh claude_codeonly_temp-1.0_p-0.9 90 98


time bash run_bash_process_batch.sh run_autojudge_all_claude_sel.sh claude_codeonly_temp-1.0_p-0.9 99 109
time bash run_bash_process_batch.sh run_autojudge_all_claude_sel.sh claude_codeonly_temp-1.0_p-0.9 110 119
time bash run_bash_process_batch.sh run_autojudge_all_claude_sel.sh claude_codeonly_temp-1.0_p-0.9 120 129
time bash run_bash_process_batch.sh run_autojudge_all_claude_sel.sh claude_codeonly_temp-1.0_p-0.9 130 139

# test (temp=0.5)
time bash run_bash_process_batch.sh run_autojudge_all_final.sh claude_codeonly_temp-0.5_p-0.9 0 4
time bash run_bash_process_batch.sh run_autojudge_all_final.sh claude_codeonly_temp-0.5_p-0.9 5 9
time bash run_bash_process_batch.sh run_autojudge_all_final.sh claude_codeonly_temp-0.5_p-0.9 10 14
time bash run_bash_process_batch.sh run_autojudge_all_final.sh claude_codeonly_temp-0.5_p-0.9 15 19
time bash run_bash_process_batch.sh run_autojudge_all_final.sh claude_codeonly_temp-0.5_p-0.9 20 26

time bash run_bash_process_batch.sh run_autojudge_all_final.sh claude_codeonly_temp-0.5_p-0.9 20 26
time bash run_bash_process_batch.sh run_autojudge_all_final.sh claude_codeonly_temp-0.5_p-0.9 20 26
time bash run_bash_process_batch.sh run_autojudge_all_final.sh claude_codeonly_temp-0.5_p-0.9 20 26


time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-0.9 27 39
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-0.9 40 49
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-0.9 50 59
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-0.9 60 69
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-0.9 70 79
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-0.9 80 89
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-0.9 90 99
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-0.9 100 109
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-0.9 110 119
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-0.9 120 129
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-0.9 130 139
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-0.9 140 149

time bash run_bash_process_batch.sh run_autojudge_all_claude_sel_2.sh claude_codeonly_temp-0.5_p-0.9 150 159
time bash run_bash_process_batch.sh run_autojudge_all_claude_sel_2.sh claude_codeonly_temp-0.5_p-0.9 160 169
time bash run_bash_process_batch.sh run_autojudge_all_claude_sel_2.sh claude_codeonly_temp-0.5_p-0.9 170 179
time bash run_bash_process_batch.sh run_autojudge_all_claude_sel_2.sh claude_codeonly_temp-0.5_p-0.9 180 189
time bash run_bash_process_batch.sh run_autojudge_all_claude_sel_2.sh claude_codeonly_temp-0.5_p-0.9 190 199


# test (temp=0.5, p=1.0)
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 0 9
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 10 19
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 20 29
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 30 39
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 40 49
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 50 59
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 60 69
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 70 79
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 80 89
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 90 99
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 100 109
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 110 119
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 120 129
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 130 139
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 140 149
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 150 159
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 160 169
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 170 179
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 180 189
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-0.5_p-1.0 190 199

# test (temp=1.0, p=1.0)
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-1.0_p-1.0 0 19
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-1.0_p-1.0 20 39
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-1.0_p-1.0 40 59
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-1.0_p-1.0 60 79
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-1.0_p-1.0 80 99
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-1.0_p-1.0 100 109
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-1.0_p-1.0 110 119
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-1.0_p-1.0 120 139
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-1.0_p-1.0 140 159
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-1.0_p-1.0 160 179
time bash run_bash_process_batch.sh run_autojudge_all_claude_final.sh claude_codeonly_temp-1.0_p-1.0 180 199



# parse (temp=1.0, p=0.9)
python parse_stats_gen_final.py --input_id claude_codeonly_temp-1.0_p-0.9 --n_gen 34

python parse_stats_gen_final.py \
    --input_id claude_codeonly_temp-1.0_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_sel.txt \
    --n_gen 140 \
    --n_probs 17 \
    --output_filename stats_claude_codeonly_temp-1.0_p-0.9_sel_17

# parse (temp=0.5, p=0.9)
python parse_stats_gen_final.py \
    --input_id claude_codeonly_temp-0.5_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --n_gen 150 \
    --n_probs 50 \
    --output_filename stats_claude_codeonly_temp-0.5_p-0.9

python parse_stats_gen_final.py \
    --input_id claude_codeonly_temp-0.5_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_sel_2.txt \
    --n_gen 200 \
    --n_probs 6 \
    --output_filename stats_claude_codeonly_temp-`0.5_p-0.9_sel_2

# parse (temp=0.5, p=1.0)
python parse_stats_gen_final.py \
    --input_id claude_codeonly_temp-0.5_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --n_gen 200 \
    --n_probs 50 \
    --output_filename stats_claude_codeonly_temp-0.5_p-1.0

# parse (temp=1.0, p=1.0)
python parse_stats_gen_final.py \
    --input_id claude_codeonly_temp-1.0_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --n_gen 200 \
    --n_probs 50 \
    --output_filename stats_claude_codeonly_temp-1.0_p-1.0

------------------------------------------------------------
# [0408] obtain indices
python obtain_claude_indices_temp-1.0_p-0.9.py
python obtain_claude_indices_temp-0.5_p-0.9.py
python obtain_claude_indices_temp-1.0_p-1.0.py
python obtain_claude_indices_temp-0.5_p-1.0.py

-> codeforces_A_gen_claude_temp-1.0_p-0.9_index.npy
-> codeforces_A_file_paths_claude_final.txt

# verify correctness of the indices
python verify_claude_correct.py
python verify_claude_correct.py \
    --input_id claude_codeonly_temp-0.5_p-0.9 \
    --idx_fileid temp-0.5_p-0.9

python write_run_autojudge_claude_final.py

------------------------------------------------------------
# generate summary
python generation_summary_gpt-3.5-instruct_traincode_gen_claude_final.py --input_id claude_codeonly_temp-1.0_p-0.9 --idx_fileid temp-1.0_p-0.9
python generation_summary_gpt-3.5-instruct_traincode_gen_claude_final.py --input_id claude_codeonly_temp-0.5_p-0.9 --idx_fileid temp-0.5_p-0.9

python generation_alg_ds_traincode_gen_claude_final_gpt-3.5-instruct.py  --input_id claude_codeonly_temp-1.0_p-0.9 --idx_fileid temp-1.0_p-0.9
python generation_alg_ds_traincode_gen_claude_final_gpt-3.5-instruct.py  --input_id claude_codeonly_temp-0.5_p-0.9 --idx_fileid temp-0.5_p-0.9

python generation_summary_gpt-3.5-instruct_traincode_gen_claude_final.py --input_id claude_codeonly_temp-1.0_p-1.0 --idx_fileid temp-1.0_p-1.0
python generation_summary_gpt-3.5-instruct_traincode_gen_claude_final.py --input_id claude_codeonly_temp-0.5_p-1.0 --idx_fileid temp-0.5_p-1.0

python generation_alg_ds_traincode_gen_claude_final_gpt-3.5-instruct.py  --input_id claude_codeonly_temp-1.0_p-1.0 --idx_fileid temp-1.0_p-1.0
python generation_alg_ds_traincode_gen_claude_final_gpt-3.5-instruct.py  --input_id claude_codeonly_temp-0.5_p-1.0 --idx_fileid temp-0.5_p-1.0


-----------
# [0425] parse generated summary 

python parse_desc_to_json_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-1.0 \
    --model gpt-3.5-instruct \
    --idx_fileid _temp-0.5_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt
python parse_desc_to_json_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-1.0 \
    --model gpt-3.5-instruct \
    --idx_fileid _temp-1.0_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt
python parse_desc_to_json_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-0.9 \
    --model gpt-3.5-instruct \
    --idx_fileid _temp-0.5_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt
python parse_desc_to_json_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-0.9 \
    --model gpt-3.5-instruct \
    --idx_fileid _temp-1.0_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt

# [0425] parse generated alg & ds
python parse_alg_ds_to_json_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-1.0 \
    --model gpt-3.5-instruct \
    --idx_fileid _temp-0.5_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt
python parse_alg_ds_to_json_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-1.0 \
    --model gpt-3.5-instruct \
    --idx_fileid _temp-1.0_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt
python parse_alg_ds_to_json_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-0.9 \
    --model gpt-3.5-instruct \
    --idx_fileid _temp-0.5_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt
python parse_alg_ds_to_json_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-0.9 \
    --model gpt-3.5-instruct \
    --idx_fileid _temp-1.0_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt

------------------------------------------------------------
# [0425] algorithms and data structures (set) similarity

python analyze_discrete_similarity_train_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-0.9 --idx_fileid _temp-1.0_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field algorithms
python analyze_discrete_similarity_train_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-0.9 --idx_fileid _temp-1.0_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field data_structures

python analyze_discrete_similarity_train_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field algorithms
python analyze_discrete_similarity_train_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field data_structures

python analyze_discrete_similarity_train_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-0.9 --idx_fileid _temp-0.5_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field algorithms
python analyze_discrete_similarity_train_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-0.9 --idx_fileid _temp-0.5_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field data_structures

python analyze_discrete_similarity_train_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field algorithms
python analyze_discrete_similarity_train_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field data_structures
-> data_structures_train_solution_similarity_gen_claude_codeonly_temp-0.5_p-1.0_gpt-3.5-instruct_final.npy

-----------
# [0425] tag similarity

python analyze_tag_similarity_train_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt
-> tags_train_solution_similarity_gen_claude_codeonly_temp-0.5_p-1.0_gpt-3.5-instruct_final.npy

python analyze_tag_similarity_train_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-0.9 --idx_fileid _temp-0.5_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt

python analyze_tag_similarity_train_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt

python analyze_tag_similarity_train_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-0.9 --idx_fileid _temp-1.0_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt

-----------
# [0425] analyze text embedding similarity

python analyze_embedding_text_sentence_transformer_gen.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-0.9 --idx_fileid _temp-1.0_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field description -d 0
python analyze_embedding_text_sentence_transformer_gen.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-0.9 --idx_fileid _temp-1.0_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field functionality -d 0
python analyze_embedding_text_sentence_transformer_gen.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-0.9 --idx_fileid _temp-1.0_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field algorithm -d 0
python analyze_embedding_text_sentence_transformer_gen.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-0.9 --idx_fileid _temp-1.0_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field data_structure -d 0
-> pairwise_sim_scores_description_gpt-3.5-instruct_gen_claude_codeonly_temp-1.0_p-0.9_final.npy

python analyze_embedding_text_sentence_transformer_gen.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field description -d 0
python analyze_embedding_text_sentence_transformer_gen.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field functionality -d 0
python analyze_embedding_text_sentence_transformer_gen.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field algorithm -d 0
python analyze_embedding_text_sentence_transformer_gen.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field data_structure -d 0
-> pairwise_sim_scores_description_gpt-3.5-instruct_gen_claude_codeonly_temp-1.0_p-1.0_final.npy

python analyze_embedding_text_sentence_transformer_gen.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field description -d 1
python analyze_embedding_text_sentence_transformer_gen.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field functionality -d 1
python analyze_embedding_text_sentence_transformer_gen.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field algorithm -d 1
python analyze_embedding_text_sentence_transformer_gen.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field data_structure -d 1
-> pairwise_sim_scores_description_gpt-3.5-instruct_gen_claude_codeonly_temp-0.5_p-1.0_final.npy


python analyze_embedding_text_sentence_transformer_gen.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-0.9 --idx_fileid _temp-0.5_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field description -d 2
python analyze_embedding_text_sentence_transformer_gen.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-0.9 --idx_fileid _temp-0.5_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field functionality -d 2
python analyze_embedding_text_sentence_transformer_gen.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-0.9 --idx_fileid _temp-0.5_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field algorithm -d 2
python analyze_embedding_text_sentence_transformer_gen.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-0.9 --idx_fileid _temp-0.5_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --field data_structure -d 2
-> pairwise_sim_scores_description_gpt-3.5-instruct_gen_claude_codeonly_temp-0.5_p-0.9_final.npy

-----------
# [0425] complexity

python analyze_complexity_entropy_train_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-0.9 --idx_fileid _temp-0.5_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt
-> complexity_train_solution_entropy_gen_final_gpt-3.5-instruct_claude_codeonly_temp-0.5_p-0.9.npz

python analyze_complexity_entropy_train_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt
python analyze_complexity_entropy_train_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt
python analyze_complexity_entropy_train_gen_final.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-0.9 --idx_fileid _temp-1.0_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt

-----------
# [0425] code efficiency

python analyze_code_efficiency_train_gen_final_gpt4.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --n_probs 50
-> run_judge_stats/stats_gen_final_claude_codeonly_temp-1.0_p-1.0.npz

python analyze_code_efficiency_train_gen_final_gpt4.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-0.9 --idx_fileid _temp-1.0_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --n_probs 50

python analyze_code_efficiency_train_gen_final_gpt4.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-0.9 --idx_fileid _temp-0.5_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --n_probs 50

python analyze_code_efficiency_train_gen_final_gpt4.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt \
    --n_probs 50

------------------------------------------------------------
# [0425] analyze accuracy

python calc_acc_train_claude_temp-1.0_p-0.9.py
python calc_acc_train_claude_temp-0.5_p-0.9.py
python calc_acc_train_claude_temp-0.5_p-1.0.py
python calc_acc_train_claude_temp-1.0_p-1.0.py


------------------------------------------------------------
# [0425] copydetect move files

python copy_train_gen_final_gpt4.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt
python copy_train_gen_final_gpt4.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-0.5_p-0.9 --idx_fileid _temp-0.5_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt
python copy_train_gen_final_gpt4.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-0.9 --idx_fileid _temp-1.0_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt
python copy_train_gen_final_gpt4.py \
    --model_fam claude \
    --input_id claude_codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt

------------------------------------------------------------
# [0425] copydetect run copydetect

python copydetect_train_gen_final_gpt4.py --model claude \
    --input_id claude_codeonly_temp-0.5_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt
python copydetect_train_gen_final_gpt4.py --model claude \
    --input_id claude_codeonly_temp-0.5_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt
python copydetect_train_gen_final_gpt4.py --model claude \
    --input_id claude_codeonly_temp-1.0_p-1.0 \
    --input_filename codeforces_A_file_paths_claude_final.txt
python copydetect_train_gen_final_gpt4.py --model claude \
    --input_id claude_codeonly_temp-1.0_p-0.9 \
    --input_filename codeforces_A_file_paths_claude_final.txt

------------------------------------------------------------
# [0425] copydetect run analyze

python analyze_copydetect_scores_final.py \
    --input_folder copydetect_scores_gen_final_claude_claude_codeonly_temp-0.5_p-1.0/ \
    --input_filename codeforces_A_file_paths_claude_final.txt
python analyze_copydetect_scores_final.py \
    --input_folder copydetect_scores_gen_final_claude_claude_codeonly_temp-0.5_p-0.9/ \
    --input_filename codeforces_A_file_paths_claude_final.txt
python analyze_copydetect_scores_final.py \
    --input_folder copydetect_scores_gen_final_claude_claude_codeonly_temp-1.0_p-1.0/ \
    --input_filename codeforces_A_file_paths_claude_final.txt
python analyze_copydetect_scores_final.py \
    --input_folder copydetect_scores_gen_final_claude_claude_codeonly_temp-1.0_p-0.9/ \
    --input_filename codeforces_A_file_paths_claude_final.txt

