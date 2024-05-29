# generation (all)
bash run_batch_generation.sh codeonly_temp-0.5 0.5 0 79

# convert (0-39)
bash run_batch_process.sh parse_gpt4_train_gen_final.py codeonly_temp-0.5 0 39

# write bash (all)
python write_bash_run_judge_all_final.py
-> run_autojudge_all_final.sh

# running eval (0-39)
time bash run_bash_process_batch.sh run_autojudge_all_final.sh codeonly_temp-0.5 0 39

# parse stats (0-39)
python parse_stats_gen_final.py --input_id codeonly_temp-0.5 --n_gen 40

# check generation till 80 complete
python check_missing_files.py -> no missing

# convert (40-79)
bash run_batch_process.sh parse_gpt4_train_gen_final.py codeonly_temp-0.5 40 79

# running eval (40-79)
time bash run_bash_process_batch.sh run_autojudge_all_final.sh codeonly_temp-0.5 40 79

python parse_stats_gen_final.py --input_id codeonly_temp-0.5 --n_gen 80
-> run_judge_stats/stats_codeonly_temp-0.5.npz


-----------
# handle the ones with < 20 correct solutions


bash run_batch_generation_alt.sh codeonly_temp-0.5 0.5 80 99

# convert (80-99)
bash run_batch_process.sh parse_gpt4_train_gen_sel_temp-0.5.py codeonly_temp-0.5 80 99

python write_bash_run_judge_all_final_sel_temp-0.5.py
-> run_autojudge_all_sel_temp-0.5.sh

time bash run_bash_process_batch.sh run_autojudge_all_sel_temp-0.5.sh codeonly_temp-0.5 80 99

python parse_stats_gen_sel_temp-0.5.py --input_id codeonly_temp-0.5 --n_gen 100
-> run_judge_stats/stats_codeonly_temp-0.5_sel_temp-0.5.npz

python verify_gpt4_correct.py --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5

other: 80; sel: 100

finally -- 
file_path in [
            'train-00001_problem_info_15.json', -- 18
            'train-00005_problem_info_13.json', -- 17
            'train-00007_problem_info_19.json', -- 13
        ]:


-----------
# generate summary

python generation_summary_traincode_gen_final.py --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5
python generation_alg_ds_traincode_gen_final_gpt-4.py --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5


python generation_summary_gpt-3.5-instruct_traincode_gen_final.py --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5
python generation_alg_ds_traincode_gen_final_gpt-3.5-instruct.py  --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5
-----------
# [0327] parse generated summary 

python parse_desc_to_json_gen_final.py --input_id codeonly_temp-0.5 --model gpt-4 --idx_fileid _temp-0.5
python parse_desc_to_json_gen_final.py --input_id codeonly_temp-0.5 --model gpt-3.5-instruct --idx_fileid _temp-0.5

# [0327] parse generated alg & ds
python parse_alg_ds_to_json_gen_final.py --input_id codeonly_temp-0.5 --model gpt-4 --idx_fileid _temp-0.5
python parse_alg_ds_to_json_gen_final.py --input_id codeonly_temp-0.5 --model gpt-3.5-instruct --idx_fileid _temp-0.5

-----------
# [0327] algorithms and data structures (set) similarity

python analyze_discrete_similarity_train_gen_final.py \
    --model gpt-4 --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5 --field algorithms

python analyze_discrete_similarity_train_gen_final.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5 --field algorithms

python analyze_discrete_similarity_train_gen_final.py \
    --model gpt-4 --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5 --field data_structures

python analyze_discrete_similarity_train_gen_final.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5 --field data_structures

-----------
# [0327] tag similarity

python analyze_tag_similarity_train_gen_final.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5
-> tags_train_solution_similarity_gen_codeonly_temp-0.5_gpt-3.5-instruct_final.npy

python analyze_tag_similarity_train_gen_final.py \
    --model gpt-4 --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5
-> tags_train_solution_similarity_gen_codeonly_temp-0.5_gpt-4_final.npy

-----------
# [0328] analyze text embedding similarity

python analyze_embedding_text_sentence_transformer_gen.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5 --field description -d 0
python analyze_embedding_text_sentence_transformer_gen.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5 --field functionality -d 1
python analyze_embedding_text_sentence_transformer_gen.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5 --field algorithm -d 2
python analyze_embedding_text_sentence_transformer_gen.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5 --field data_structure -d 3

-> pairwise_sim_scores_description_gpt-3.5-instruct_gen_codeonly_temp-0.5_final.npy
-> pairwise_sim_scores_functionality_gpt-3.5-instruct_gen_codeonly_temp-0.5_final.npy
-> pairwise_sim_scores_algorithm_gpt-3.5-instruct_gen_codeonly_temp-0.5_final.npy
-> pairwise_sim_scores_data_structure_gpt-3.5-instruct_gen_codeonly_temp-0.5_final.npy

python analyze_embedding_text_sentence_transformer_gen.py \
    --model gpt-4 --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5 --field description -d 0
python analyze_embedding_text_sentence_transformer_gen.py \
    --model gpt-4 --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5 --field functionality -d 1
python analyze_embedding_text_sentence_transformer_gen.py \
    --model gpt-4 --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5 --field algorithm -d 2
python analyze_embedding_text_sentence_transformer_gen.py \
    --model gpt-4 --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5 --field data_structure -d 3

-> pairwise_sim_scores_description_gpt-4_gen_codeonly_temp-0.5_final.npy
-> pairwise_sim_scores_functionality_gpt-4_gen_codeonly_temp-0.5_final.npy
-> pairwise_sim_scores_algorithm_gpt-4_gen_codeonly_temp-0.5_final.npy
-> pairwise_sim_scores_data_structure_gpt-4_gen_codeonly_temp-0.5_final.npy

-----------
# [0328] complexity

python analyze_complexity_entropy_train_gen_final.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5
-> complexity_train_solution_entropy_gen_final_gpt-3.5-instruct_codeonly_temp-0.5.npz

python analyze_complexity_entropy_train_gen_final.py \
    --model gpt-4 --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5
-> complexity_train_solution_entropy_gen_final_gpt-4_codeonly_temp-0.5.npz

-----------
# [0328] code efficiency

python analyze_code_efficiency_train_gen_final_gpt4.py \
    --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5
