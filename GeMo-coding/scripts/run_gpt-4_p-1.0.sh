# generation (all)
bash run_batch_generation_p.sh codeonly_temp-0.5_p-1.0 0.5 1.0 0 79
bash run_batch_generation_p.sh codeonly_temp-1.0_p-1.0 1.0 1.0 0 79

bash run_batch_generation_p.sh codeonly_temp-0.5_p-1.0 0.5 1.0 80 99
bash run_batch_generation_p.sh codeonly_temp-1.0_p-1.0 1.0 1.0 80 84

# convert (0-39)
bash run_batch_process.sh parse_gpt4_train_gen_final.py codeonly_temp-0.5_p-1.0 0 39
bash run_batch_process.sh parse_gpt4_train_gen_final.py codeonly_temp-1.0_p-1.0 0 39

bash run_batch_process.sh parse_gpt4_train_gen_final.py codeonly_temp-0.5_p-1.0 40 59
bash run_batch_process.sh parse_gpt4_train_gen_final.py codeonly_temp-1.0_p-1.0 40 59

bash run_batch_process.sh parse_gpt4_train_gen_final.py codeonly_temp-0.5_p-1.0 60 79
bash run_batch_process.sh parse_gpt4_train_gen_final.py codeonly_temp-1.0_p-1.0 60 79

bash run_batch_process.sh parse_gpt4_train_gen_final.py codeonly_temp-0.5_p-1.0 80 94
bash run_batch_process.sh parse_gpt4_train_gen_final.py codeonly_temp-0.5_p-1.0 95 99
bash run_batch_process.sh parse_gpt4_train_gen_final.py codeonly_temp-1.0_p-1.0 80 84

# [0329] run eval (0-59)
time bash run_bash_process_batch.sh run_autojudge_all_final.sh codeonly_temp-0.5_p-1.0 0 39
time bash run_bash_process_batch.sh run_autojudge_all_final.sh codeonly_temp-1.0_p-1.0 0 39

time bash run_bash_process_batch.sh run_autojudge_all_final.sh codeonly_temp-0.5_p-1.0 40 59
time bash run_bash_process_batch.sh run_autojudge_all_final.sh codeonly_temp-1.0_p-1.0 40 59

time bash run_bash_process_batch.sh run_autojudge_all_final.sh codeonly_temp-0.5_p-1.0 60 79
time bash run_bash_process_batch.sh run_autojudge_all_final.sh codeonly_temp-1.0_p-1.0 60 79

time bash run_bash_process_batch.sh run_autojudge_all_final.sh codeonly_temp-0.5_p-1.0 80 94
time bash run_bash_process_batch.sh run_autojudge_all_final.sh codeonly_temp-0.5_p-1.0 95 99
time bash run_bash_process_batch.sh run_autojudge_all_final.sh codeonly_temp-1.0_p-1.0 80 84

# parse stats (0-79)
python parse_stats_gen_final.py --input_id codeonly_temp-0.5_p-1.0 --n_gen 100
-> run_judge_stats/stats_codeonly_temp-0.5_p-1.0.npz
python parse_stats_gen_final.py --input_id codeonly_temp-1.0_p-1.0 --n_gen 85
-> run_judge_stats/stats_codeonly_temp-1.0_p-1.0.npz

-----------
# [0330, 0331] obtain indices
python obtain_indices.py
python obtain_indices_t-0.5_p-1.0.py

-----------
# [0330, 0331] generate summary & alg ds
python generation_summary_gpt-3.5-instruct_traincode_gen_final.py --input_id codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0
python generation_alg_ds_traincode_gen_final_gpt-3.5-instruct.py  --input_id codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0

python generation_summary_gpt-3.5-instruct_traincode_gen_final.py --input_id codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0
python generation_alg_ds_traincode_gen_final_gpt-3.5-instruct.py  --input_id codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0

-----------
# [0331] parse generated summary and alg ds

python parse_desc_to_json_gen_final.py --input_id codeonly_temp-0.5_p-1.0 --model gpt-3.5-instruct --idx_fileid _temp-0.5_p-1.0
python parse_alg_ds_to_json_gen_final.py --input_id codeonly_temp-0.5_p-1.0 --model gpt-3.5-instruct --idx_fileid _temp-0.5_p-1.0
python parse_desc_to_json_gen_final.py --input_id codeonly_temp-1.0_p-1.0 --model gpt-3.5-instruct --idx_fileid _temp-1.0_p-1.0
python parse_alg_ds_to_json_gen_final.py --input_id codeonly_temp-1.0_p-1.0 --model gpt-3.5-instruct --idx_fileid _temp-1.0_p-1.0

########################################################
##############  temp-0.5_p-1.0
########################################################

# [0331] algorithms and data structures (set) similarity

python analyze_discrete_similarity_train_gen_final.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0 --field algorithms
-> algorithms_train_solution_similarity_gen_codeonly_temp-0.5_p-1.0_gpt-3.5-instruct_final.npy

python analyze_discrete_similarity_train_gen_final.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0 --field data_structures
-> data_structures_train_solution_similarity_gen_codeonly_temp-0.5_p-1.0_gpt-3.5-instruct_final.npy


python analyze_discrete_similarity_train_gen_final.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0 --field algorithms
-> algorithms_train_solution_similarity_gen_codeonly_temp-1.0_p-1.0_gpt-3.5-instruct_final.npy

python analyze_discrete_similarity_train_gen_final.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0 --field data_structures
-> data_structures_train_solution_similarity_gen_codeonly_temp-1.0_p-1.0_gpt-3.5-instruct_final.npy

-----------
# [0331] tag similarity
python analyze_tag_similarity_train_gen_final.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0
-> tags_train_solution_similarity_gen_codeonly_temp-0.5_p-1.0_gpt-3.5-instruct_final.npy

python analyze_tag_similarity_train_gen_final.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0
-> tags_train_solution_similarity_gen_codeonly_temp-1.0_p-1.0_gpt-3.5-instruct_final.npy

-----------
# [0331] analyze text embedding similarity

python analyze_embedding_text_sentence_transformer_gen.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0 --field description -d 0
python analyze_embedding_text_sentence_transformer_gen.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0 --field functionality -d 1
python analyze_embedding_text_sentence_transformer_gen.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0 --field algorithm -d 2
python analyze_embedding_text_sentence_transformer_gen.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0 --field data_structure -d 3

-> pairwise_sim_scores_description_gpt-3.5-instruct_gen_codeonly_temp-0.5_p-1.0_final.npy
-> pairwise_sim_scores_functionality_gpt-3.5-instruct_gen_codeonly_temp-0.5_p-1.0_final.npy
-> pairwise_sim_scores_algorithm_gpt-3.5-instruct_gen_codeonly_temp-0.5_p-1.0_final.npy
-> pairwise_sim_scores_data_structure_gpt-3.5-instruct_gen_codeonly_temp-0.5_p-1.0_final.npy


python analyze_embedding_text_sentence_transformer_gen.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0 --field description -d 0
python analyze_embedding_text_sentence_transformer_gen.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0 --field functionality -d 1
python analyze_embedding_text_sentence_transformer_gen.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0 --field algorithm -d 2
python analyze_embedding_text_sentence_transformer_gen.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0 --field data_structure -d 3

-> pairwise_sim_scores_description_gpt-3.5-instruct_gen_codeonly_temp-1.0_p-1.0_final.npy
-> pairwise_sim_scores_functionality_gpt-3.5-instruct_gen_codeonly_temp-1.0_p-1.0_final.npy
-> pairwise_sim_scores_algorithm_gpt-3.5-instruct_gen_codeonly_temp-1.0_p-1.0_final.npy
-> pairwise_sim_scores_data_structure_gpt-3.5-instruct_gen_codeonly_temp-1.0_p-1.0_final.npy

-----------
# [0331] complexity

python analyze_complexity_entropy_train_gen_final.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0
-> complexity_train_solution_entropy_gen_final_gpt-3.5-instruct_codeonly_temp-0.5_p-1.0.npz

python analyze_complexity_entropy_train_gen_final.py \
    --model gpt-3.5-instruct --input_id codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0
-> complexity_train_solution_entropy_gen_final_gpt-3.5-instruct_codeonly_temp-1.0_p-1.0.npz

-----------
# [0331] code efficiency

python analyze_code_efficiency_train_gen_final_gpt4.py \
    --input_id codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0
-> run_judge_stats/stats_gen_final_codeonly_temp-0.5_p-1.0.npz

python analyze_code_efficiency_train_gen_final_gpt4.py \
    --input_id codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0
-> run_judge_stats/stats_gen_final_codeonly_temp-1.0_p-1.0.npz

-----------
# [0331] accuracy
python calc_acc_train_gpt4_general.py --input_id codeonly_temp-0.5_p-1.0
python calc_acc_train_gpt4_general.py --input_id codeonly_temp-1.0_p-1.0
TODO: mann whitney

-----------
# [0331] prepare for copydetect (copy files)
python copy_train_gen_final_gpt4.py --input_id codeonly_temp-0.5_p-1.0 --idx_fileid _temp-0.5_p-1.0
-> train_gen_final_gpt4_codeonly_temp-0.5_p-1.0
python copy_train_gen_final_gpt4.py --input_id codeonly_temp-1.0_p-1.0 --idx_fileid _temp-1.0_p-1.0
-> train_gen_final_gpt4_codeonly_temp-1.0_p-1.0

------------------------------------------------------------------
# [0331] run copydetect
python copydetect_train_gen_final_gpt4.py --input_id codeonly_temp-0.5_p-1.0
python analyze_copydetect_scores_final.py --input_folder copydetect_scores_gen_final_gpt4_codeonly_temp-0.5_p-1.0
-> copydetect_scores_gen_final_gpt4_codeonly_temp-0.5_p-1.0/all_scores.npy

python copydetect_train_gen_final_gpt4.py --input_id codeonly_temp-1.0_p-1.0
python analyze_copydetect_scores_final.py --input_folder copydetect_scores_gen_final_gpt4_codeonly_temp-1.0_p-1.0
-> copydetect_scores_gen_final_gpt4_codeonly_temp-1.0_p-1.0/all_scores.npy
