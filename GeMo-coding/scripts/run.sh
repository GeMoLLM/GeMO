bazel run -c opt   :print_names_and_sources /home/fanw6/main/dm-code_contests/code_contests_valid.riegeli > out_info.txt

bazel run -c opt execution:solve_example -- \
  --valid_path=/home/fanw6/main/dm-code_contests/code_contests_valid.riegeli

bazel run -c opt execution:solve_example_10 --   \
  --valid_path=/home/fanw6/main/dm-code_contests/code_contests_valid.riegeli \
  --problem_name="1551_B1. Wonderful Coloring - 1"


bazel run -c opt execution:solve_example_10 --   \
  --valid_path=/home/fanw6/main/dm-code_contests/code_contests_valid.riegeli \
  --problem_name="1549_A. Gregor and Cryptography" \
  --solution_path="valid_problem_0_correct_0.txt"

bazel run -c opt execution:solve_example_10 --   \
  --valid_path=/home/fanw6/main/dm-code_contests/code_contests_train.riegeli-00000-of-00128 \
  --problem_name="1000_A. Codehorses T-shirts" \
  --solution_path="train-00000_0_solutions_0.txt"


python convert_code.py --input_file valid_problem_0.json

bash run_convert_all.sh # convert the json code to the txt code



python write_bash.py # write the bash file to run all the judge

bash run_bazel_all.sh

python write_bash_gpt4_gen_1.py
python write_bash_gpt4_gen_2.py
python write_bash_gpt4_gen.py --input_prefix valid_solution_persona_codeonly \
  --outfile_id gpt4_gen_3
python write_bash_gpt4_gen.py --input_prefix valid_solution_persona_stepbystep \
  --outfile_id gpt4_gen_4
python write_bash_gpt4_selgen.py --input_prefix valid_solution_codeonly_1 \
  --outfile_id gpt4_gen_codeonly_1


bash run_bazel_all_gpt4_gen_1.sh
bash run_bazel_all_gpt4_gen_2.sh
bash run_bazel_all_gpt4_gen.sh gpt4_gen_3
bash run_bazel_all_gpt4_gen.sh gpt4_gen_4
bash run_bazel_all_gpt4_selgen.sh gpt4_gen_codeonly_1

- output format: line 1 (compile ok?); line 2 (pass how many tests); line 3 (duration)

# loc
python count_code_loc.py

# accuracy, duration
python agg_judge_output.py

# dataset accuracy
python examine_acc.py

python language_entropy.py
python language_python_portion.py



python generation_code_gpt4.py

-> valid_solution_{i}.txt
-> valid_solution_codeonly_{i}.txt
-> valid_solution_persona_codeonly_{i}.txt
-> valid_solution_persona_stepbystep_{i}.txt

python parse_gpt4_gen_1.py
python parse_gpt4_gen_1.py --input_prefix valid_solution_codeonly
python parse_gpt4_gen_1.py --input_prefix valid_solution_persona_codeonly
python parse_gpt4_gen_1.py --input_prefix valid_solution_persona_stepbystep
python parse_gpt4_gen_multiple.py --input_prefix valid_solution_persona_stepbystep

python analyze_gpt4_gen_acc.py --input_prefix valid_solution_persona_codeonly \
  --outfile_id gpt4_gen_3

python analyze_gpt4_gen_acc.py --input_prefix valid_solution_persona_stepbystep \
  --outfile_id gpt4_gen_4

python analyze_gpt4_selgen_acc.py --input_prefix valid_solution_codeonly_1 \
  --outfile_id gpt4_gen_codeonly_1

bazel run -c opt   :print_names_and_sources /home/fanw6/main/dm-code_contests/code_contests_valid.riegeli > out_info.txt


python generation_selcode_gpt4.py --output_id codeonly_1 --temperature 1.0;\
python generation_selcode_gpt4.py --output_id codeonly_2 --temperature 1.0;\
python generation_selcode_gpt4.py --output_id codeonly_3 --temperature 1.0;\
python generation_selcode_gpt4.py --output_id codeonly_4 --temperature 1.0;\
python generation_selcode_gpt4.py --output_id codeonly_5 --temperature 1.0;\


python parse_gpt4_selgen.py --input_prefix valid_solution_codeonly_1
python parse_gpt4_selgen.py --input_prefix valid_solution_codeonly_2


bash run_eval.sh codeonly_2
bash run_eval.sh codeonly_3
bash run_eval.sh codeonly_4
bash run_eval.sh codeonly_5


python analyze_mul_acc.py 


python generation_summary_selcode_gpt4.py --input_id codeonly_1
python generation_summary_selcode_gpt4.py --input_id codeonly_2
python generation_summary_selcode_gpt4.py --input_id codeonly_3
python generation_summary_selcode_gpt4.py --input_id codeonly_4
python generation_summary_selcode_gpt4.py --input_id codeonly_5


python format_check_descriptions.py
python parse_descriptions.py

------------------------------------------------------------------
python tags.py

python generation_tags_selcode_gpt4.py --input_id codeonly_1;\
python generation_tags_selcode_gpt4.py --input_id codeonly_2;\
python generation_tags_selcode_gpt4.py --input_id codeonly_3;\
python generation_tags_selcode_gpt4.py --input_id codeonly_4;\
python generation_tags_selcode_gpt4.py --input_id codeonly_5

python analyze_tag_diversity.py
# tags_valid_solution_{args.input_id}_diversity.npy

------------------------------------------------------------------
# date: 0311
# obtain train info

bazel run -c opt   :print_names_and_sources /home/fanw6/main/dm-code_contests/code_contests_train.riegeli-00000-of-00128 train-00000

bash run_extract_train_data.sh

------------------------------------------------------------------

# filter train file ids by codeforces rating A problems

python filter_codeforces_A_index.py

-> codeforces_A_file_paths.txt

------------------------------------------------------------------
# generating code
python generation_train_code_gpt4.py --output_id codeonly_1 --temperature 1.0
python generation_train_code_gpt4.py --output_id codeonly_0 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_2 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_3 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_4 --temperature 1.0;\

python generation_train_code_gpt4.py --output_id codeonly_5 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_6 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_7 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_8 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_9 --temperature 1.0;\

python generation_train_code_gpt4.py --output_id codeonly_10 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_11 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_12 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_13 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_14 --temperature 1.0;\

python generation_train_code_gpt4.py --output_id codeonly_15 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_16 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_17 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_18 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_19 --temperature 1.0;\

python generation_train_code_gpt4.py --output_id codeonly_20 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_21 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_22 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_23 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_24 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_25 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_26 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_27 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_28 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_29 --temperature 1.0;\

python generation_train_code_gpt4.py --output_id codeonly_30 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_31 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_32 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_33 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_34 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_35 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_36 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_37 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_38 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_39 --temperature 1.0;\

python generation_train_code_gpt4.py --output_id codeonly_40 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_41 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_42 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_43 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_44 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_45 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_46 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_47 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_48 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_49 --temperature 1.0;\

python generation_train_code_gpt4.py --output_id codeonly_50 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_51 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_52 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_53 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_54 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_55 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_56 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_57 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_58 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_59 --temperature 1.0;\

python generation_train_code_gpt4.py --output_id codeonly_60 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_61 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_62 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_63 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_64 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_65 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_66 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_67 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_68 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_69 --temperature 1.0;\

python generation_train_code_gpt4.py --output_id codeonly_70 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_71 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_72 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_73 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_74 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_75 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_76 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_77 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_78 --temperature 1.0;\
python generation_train_code_gpt4.py --output_id codeonly_79 --temperature 1.0;\

# generating codeonly_{0..39} for files from 100:135
bash run_batch_generation_gpt4.sh codeonly 1.0 0 39 100 135
bash run_batch_generation_gpt4.sh codeonly 1.0 40 79 100 135

------------------------------------------------------------------
# convert train code
python convert_train_code.py --input_file train-00000_problem_info_0.json

python write_bash_convert_train_code.py
-> run_convert_train_code_all.sh

bash run_convert_train_code_all.sh

------------------------------------------------------------------
# parse generated code

python parse_gpt4_train_gen.py --input_id codeonly_0
python parse_gpt4_train_gen.py --input_id codeonly_1
python parse_gpt4_train_gen.py --input_id codeonly_2
python parse_gpt4_train_gen.py --input_id codeonly_3
python parse_gpt4_train_gen.py --input_id codeonly_4

python parse_gpt4_train_gen.py --input_id codeonly_5;\
python parse_gpt4_train_gen.py --input_id codeonly_6;\
python parse_gpt4_train_gen.py --input_id codeonly_7;\
python parse_gpt4_train_gen.py --input_id codeonly_8;\
python parse_gpt4_train_gen.py --input_id codeonly_9

bash run_batch_process.sh parse_gpt4_train_gen.py codeonly 10 14
bash run_batch_process.sh parse_gpt4_train_gen.py codeonly 15 19
bash run_batch_process.sh parse_gpt4_train_gen.py codeonly 20 79


bash run_batch_process.sh parse_gpt4_train_gen.py codeonly 0 79


------------------------------------------------------------------
# extracting information from train code
python generation_summary_traincode_src.py
python generation_alg_ds_traincode_src.py

------------------------------------------------------------------
# extracting information from generated code

## summary
python generation_summary_traincode_gpt4.py --input_id codeonly_0;\ 
python generation_summary_traincode_gpt4.py --input_id codeonly_1;\ 
python generation_summary_traincode_gpt4.py --input_id codeonly_2;\ 
python generation_summary_traincode_gpt4.py --input_id codeonly_3;\
python generation_summary_traincode_gpt4.py --input_id codeonly_4;\

python generation_summary_traincode_gpt4.py --input_id codeonly_5;\
python generation_summary_traincode_gpt4.py --input_id codeonly_6;\
python generation_summary_traincode_gpt4.py --input_id codeonly_7;\
python generation_summary_traincode_gpt4.py --input_id codeonly_8;\
python generation_summary_traincode_gpt4.py --input_id codeonly_9;\
bash run_batch_process.sh generation_summary_traincode_gpt4.py codeonly 10 14
bash run_batch_process.sh generation_summary_traincode_gpt4.py codeonly 15 19

## alg_ds
python generation_alg_ds_traincode_gpt4.py --input_id codeonly_0
bash run_batch_process.sh generation_alg_ds_traincode_gpt4.py codeonly 1 14
bash run_batch_process.sh generation_alg_ds_traincode_gpt4.py codeonly 15 19

------------------------------------------------------------------
# sanity check on the generated summary format
## for the source
python format_check_descriptions_train_src.py
python parse_desc_to_json_src.py

## for the generations

bash run_batch_process.sh parse_desc_to_json_gpt4.py codeonly 0 19

------------------------------------------------------------------
# Moss for detecting plagiarism

## one simple exp
./moss -l python train-00000_solution_codeonly_0_0.py train-00000_soluti
on_codeonly_1_0.py > moss_train-00000_solution_codeonly_0_codeonly_1_0.txt

## for the source files
python write_bash_moss_traincode_src.py
bash run_moss_traincode_src_train-00000_0.sh

python write_bash_moss_traincode_src_all.py
bash run_moss_traincode_src_all.sh

------------------------------------------------------------------
# test accuracy for generated train code

python write_bash_gpt4_traingen.py --input_id codeonly_1
python write_bash_gpt4_traingen_all.py --input_id codeonly_1

bash run_bazel_train_gpt4_codeonly_1_all.sh

bash write_bash_e2e_gpt4_traingen.sh codeonly_0

bash run_bazel_train_gpt4_codeonly_0_all.sh


python write_bash_gpt4_traingen.py
python write_bash_gpt4_traingen_all.py

bash run_bazel_train_gpt4_all.sh codeonly_1
bash run_bazel_train_gpt4_all.sh codeonly_0
bash run_bazel_train_gpt4_all.sh codeonly_2

bash run_bazel_train_gpt4_all.sh codeonly_3;\
bash run_bazel_train_gpt4_all.sh codeonly_4;\
bash run_bazel_train_gpt4_all.sh codeonly_5;\
bash run_bazel_train_gpt4_all.sh codeonly_6;\
bash run_bazel_train_gpt4_all.sh codeonly_7;\
bash run_bazel_train_gpt4_all.sh codeonly_8;\
bash run_bazel_train_gpt4_all.sh codeonly_9;\


bash run_bazel_train_gpt4_all.sh codeonly_10;\
bash run_bazel_train_gpt4_all.sh codeonly_11;\
bash run_bazel_train_gpt4_all.sh codeonly_12;\
bash run_bazel_train_gpt4_all.sh codeonly_13;\
bash run_bazel_train_gpt4_all.sh codeonly_14;\
bash run_bazel_train_gpt4_all.sh codeonly_15;\
bash run_bazel_train_gpt4_all.sh codeonly_16;\
bash run_bazel_train_gpt4_all.sh codeonly_17;\
bash run_bazel_train_gpt4_all.sh codeonly_18;\
bash run_bazel_train_gpt4_all.sh codeonly_19;\

bash run_bazel_train_gpt4_all.sh codeonly_20;\
bash run_bazel_train_gpt4_all.sh codeonly_21;\
bash run_bazel_train_gpt4_all.sh codeonly_22;\
bash run_bazel_train_gpt4_all.sh codeonly_23;\
bash run_bazel_train_gpt4_all.sh codeonly_24;\
bash run_bazel_train_gpt4_all.sh codeonly_25;\
bash run_bazel_train_gpt4_all.sh codeonly_26;\
bash run_bazel_train_gpt4_all.sh codeonly_27;\
bash run_bazel_train_gpt4_all.sh codeonly_28;\
bash run_bazel_train_gpt4_all.sh codeonly_29;\
bash run_bazel_train_gpt4_all.sh codeonly_30;\
bash run_bazel_train_gpt4_all.sh codeonly_31;\
bash run_bazel_train_gpt4_all.sh codeonly_32;\
bash run_bazel_train_gpt4_all.sh codeonly_33;\
bash run_bazel_train_gpt4_all.sh codeonly_34;\
bash run_bazel_train_gpt4_all.sh codeonly_35;\
bash run_bazel_train_gpt4_all.sh codeonly_36;\
bash run_bazel_train_gpt4_all.sh codeonly_37;\
bash run_bazel_train_gpt4_all.sh codeonly_38;\
bash run_bazel_train_gpt4_all.sh codeonly_39;\

------------------------------------------------------------------
# copy files

## for train src
python copy_train_src.py

## for train gpt4
python copy_train_gen.py --input_id codeonly



------------------------------------------------------------------
# copydetect

## for train src
python copydetect_train_src.py
python analyze_copydetect_scores.py --input_folder copydetect_scores_src
-> copydetect_scores_src/all_scores.npy
## for train gen
python copydetect_train_gen.py
python analyze_copydetect_scores.py --input_folder copydetect_scores_gen
-> copydetect_scores_gen/all_scores.npy

------------------------------------------------------------------
# tag similarity

## for train src
python analyze_tag_similarity_train_src.py
-> tags_train_solution_similarity.npy
## for train gen
python analyze_tag_similarity_train_gen.py
-> tags_train_solution_similarity_gen_codeonly.npy

------------------------------------------------------------------
# (time and space) complexity similarity
## for train src
python analyze_complexity_entropy_train_src.py
-> complexity_train_solution_entropy.npz

## for train gen
python analyze_complexity_entropy_train_gen.py
-> complexity_train_solution_entropy_gen_codeonly.npz

## comparing complexity
python analyze_complexity_mannwhitney_train.py

------------------------------------------------------------------
python analyze_judge_output_train_gpt4.py


------------------------------------------------------------------

bazel run -c opt execution:document_test_cases --   \
  --valid_path=/home/fanw6/main/dm-code_contests/code_contests_train.riegeli-00000-of-00128 \
  --problem_name="1000_A. Codehorses T-shirts"


python write_bash_train_testcases.py
python write_bash_train_testcases_all.py

bash run_bazel_document_test_cases_all.sh


------------------------------------------------------------------
bash run_judge.sh train-00000 0 1000_A._Codehorses_T-shirts codeonly_0

python write_bash_run_judge.py
python write_bash_run_judge_all.py

bash run_autojudge_all.sh codeonly_0

time bash run_bash_process_batch.sh run_autojudge_all.sh codeonly 2 79
real    306m46.415s
user    114m48.354s
sys     70m45.786s

time bash run_bash_process_batch.sh run_autojudge_all.sh codeonly 1 1

# for 100:135
time bash run_bash_process_batch.sh run_autojudge_all.sh codeonly 0 79


------------------------------------------------------------------
# why only 94? find out the missing 6!
python find_missing.py
-> 
train-00004 13 stats_train-00004_13_gen_codeonly_0.txt
train-00004 25 stats_train-00004_25_gen_codeonly_0.txt
train-00007 0 stats_train-00007_0_gen_codeonly_0.txt
train-00007 29 stats_train-00007_29_gen_codeonly_0.txt
train-00008 26 stats_train-00008_26_gen_codeonly_0.txt
train-00010 7 stats_train-00010_7_gen_codeonly_0.txt

reason is \' in the problem name

time bash run_bash_process_batch.sh run_autojudge_missing_6_of_100.sh codeonly 0 79

bash run_bash_process_batch.sh run_autojudge_train-00006_23.sh codeonly 0 79

------------------------------------------------------------------

# parse the results -> rewrite analyze_judge_output 
#  -> get accuracy results

python parse_stats.py

------------------------------------------------------------------

python write_bash_run_judge_src.py
python write_bash_run_judge_src_all.py
bash run_bash_process_batch_src.sh run_autojudge_src_all.sh 0 19

python parse_stats_src.py
------------------------------------------------------------------

bash run_extract_train_data_sel.sh


# convert train code
python convert_train_code.py --input_file train-00000_problem_info_0.json

# for the selected files (# source correct \in [17, 19]), 
# sample more solutions (25) via print_name_and_sources,
# and run evaluation on them

python write_bash_convert_train_code_sel.py
bash run_convert_train_code_all_sel.sh


python write_bash_run_judge_src_all_sel.py
bash run_bash_process_batch_src.sh run_autojudge_src_all_sel.sh 0 24

python parse_stats_src_sel.py

------------------------------------------------------------------

codeforces_A_file_paths_final.txt
codeforces_A_train_index.npy

# verify the correctness of the selected files

python verify_train_correct.py
python verify_gpt4_correct.py

------------------------------------------------------------------

# accuracy results

python calc_acc_train_gpt4.py
-> acc_train_gpt4_codeonly.npy

python calc_acc_train_gpt4.py --input_id codeonly_temp-0.5_p-1.0

python calc_acc_train_gpt4_temp-0.5.py
-> acc_train_gpt4_codeonly_temp-0.5.npy

python calc_acc_train_src.py
-> acc_train_src.npy

------------------------------------------------------------------
# copy files (according to the indices)

## for train src
python copy_train_src_final.py

## for train gpt4
python copy_train_gen_final_gpt4.py --input_id codeonly

## for train gpt4 (temp=0.5)
python copy_train_gen_final_gpt4.py --input_id codeonly_temp-0.5 --idx_fileid _temp-0.5
-> train_gen_final_gpt4_codeonly_temp-0.5

------------------------------------------------------------------
# copydetect (according to the indices)

## for train src
python copydetect_train_src_final.py

python analyze_copydetect_scores_final.py --input_folder copydetect_scores_src_final
-> copydetect_scores_src_final/all_scores.npy
## for train gen
python copydetect_train_gen_final_gpt4.py

python analyze_copydetect_scores_final.py --input_folder copydetect_scores_gen_final_gpt4
-> copydetect_scores_gen_final_gpt4/all_scores.npy

## for train gen (temp=0.5)
python copydetect_train_gen_final_gpt4.py --input_id codeonly_temp-0.5

python analyze_copydetect_scores_final.py --input_folder copydetect_scores_gen_final_gpt4_codeonly_temp-0.5
-> copydetect_scores_gen_final_gpt4_codeonly_temp-0.5/all_scores.npy

------------------------------------------------------------------
# code efficiency -- runtime and memory (according to the indices)

## for train src
python analyze_code_efficiency_train_src_final.py
-> run_judge_stats/src_stats_final.npz

python analyze_code_efficiency_train_gen_final_gpt4.py
-> run_judge_stats/stats_gen_final_codeonly.npz

------------------------------------------------------------------
# generate summaries / descriptions

# pane=3
python generation_summary_traincode_src_final.py

# done
python generation_summary_gpt-3.5-instruct_traincode_src_final.py

# pane=-3
python generation_summary_traincode_gen_final.py

# done
python generation_summary_gpt-3.5-instruct_traincode_gen_final.py

------------------------------------------------------------------
# parse generated summarization by gpt-3.5-instruct
python parse_desc_to_json_src_final.py --model gpt-3.5-instruct
python parse_desc_to_json_gen_final.py --model gpt-3.5-instruct

python parse_desc_to_json_src_final.py --model gpt4
python parse_desc_to_json_gen_final.py --model gpt4

------------------------------------------------------------------
# (time and space) complexity similarity (according to the indices)
## for train src
python analyze_complexity_entropy_train_src_final.py --model gpt-3.5-instruct
-> complexity_train_solution_entropy_gpt-3.5-instruct_final.npz

## for train gen
python analyze_complexity_entropy_train_gen_final.py --model gpt-3.5-instruct
-> complexity_train_solution_entropy_gen_final_gpt-3.5-instruct_codeonly.npz

## [0328] for train src [GPT-4 annotations]
python analyze_complexity_entropy_train_src_final.py --model gpt-4
-> complexity_train_solution_entropy_gpt-4_final.npz

## comparing complexity
python analyze_complexity_mannwhitney_train.py
-> complexity_train_solution_cmp_final_src_gen_gpt-3.5-instruct_codeonly.npz

------------------------------------------------------------------
# tag similarity (according to the indices)

## for train src
python analyze_tag_similarity_train_src_final.py --model gpt-3.5-instruct
-> tags_train_solution_similarity_gpt-3.5-instruct_final.npy

## [0328] for train src [GPT-4 annotations]
# TODO: really need to clean up for this
python analyze_tag_similarity_train_src_final.py --model gpt-4
-> tags_train_solution_similarity_gpt-4_final.npy

## for train gen
python analyze_tag_similarity_train_gen_final.py --model gpt-3.5-instruct
-> tags_train_solution_similarity_gen_codeonly_gpt-3.5-instruct_final.npy

------------------------------------------------------------------

python analyze_embedding_text_sentence_transformer_src.py --field description
-> pairwise_sim_scores_description_gpt-3.5-instruct_src_final.npy
python analyze_embedding_text_sentence_transformer_gen.py --field description
-> pairwise_sim_scores_description_gpt-3.5-instruct_gen_codeonly_final.npy

python analyze_embedding_text_sentence_transformer_src.py --field functionality
-> pairwise_sim_scores_functionality_gpt-3.5-instruct_src_final.npy
python analyze_embedding_text_sentence_transformer_gen.py --field functionality
-> pairwise_sim_scores_functionality_gpt-3.5-instruct_gen_codeonly_final.npy

python analyze_embedding_text_sentence_transformer_src.py --field data_structure
-> pairwise_sim_scores_data_structure_gpt-3.5-instruct_src_final.npy
python analyze_embedding_text_sentence_transformer_gen.py --field data_structure
-> pairwise_sim_scores_data_structure_gpt-3.5-instruct_gen_codeonly_final.npy

python analyze_embedding_text_sentence_transformer_src.py --field algorithm
-> pairwise_sim_scores_algorithm_gpt-3.5-instruct_src_final.npy
python analyze_embedding_text_sentence_transformer_gen.py --field algorithm
-> pairwise_sim_scores_algorithm_gpt-3.5-instruct_gen_codeonly_final.npy

## [0328] for train src [GPT-4 annotations]

python analyze_embedding_text_sentence_transformer_src.py --field description --model gpt-4
python analyze_embedding_text_sentence_transformer_src.py --field functionality --model gpt-4
python analyze_embedding_text_sentence_transformer_src.py --field data_structure --model gpt-4
python analyze_embedding_text_sentence_transformer_src.py --field algorithm --model gpt-4

-> pairwise_sim_scores_description_gpt-4_src_final.npy
-> pairwise_sim_scores_functionality_gpt-4_src_final.npy
-> pairwise_sim_scores_data_structure_gpt-4_src_final.npy
-> pairwise_sim_scores_algorithm_gpt-4_src_final.npy

python analyze_embedding_text_sentence_transformer_gen.py --field description --model gpt-4
python analyze_embedding_text_sentence_transformer_gen.py --field functionality --model gpt-4
python analyze_embedding_text_sentence_transformer_gen.py --field data_structure --model gpt-4
python analyze_embedding_text_sentence_transformer_gen.py --field algorithm --model gpt-4

-> pairwise_sim_scores_description_gpt-4_gen_codeonly_final.npy

------------------------------------------------------------------
# GPT-generated alg & ds

python generation_alg_ds_traincode_src_final_gpt-3.5-instruct.py 
python generation_alg_ds_traincode_gen_final_gpt-3.5-instruct.py 

# parsing

python parse_alg_ds_to_json_src_final.py --model gpt-3.5-instruct
python parse_alg_ds_to_json_gen_final.py --model gpt-3.5-instruct

python parse_alg_ds_to_json_src_final.py --model gpt-4
python parse_alg_ds_to_json_gen_final.py --model gpt-4

------------------------------------------------------------------
# algorithms and data structures (set) similarity (according to the indices)

python analyze_discrete_similarity_train_src_final.py --field algorithms
-> algorithms_train_solution_similarity_gpt-3.5-instruct_final.npy
python analyze_discrete_similarity_train_src_final.py --field data_structures
-> data_structures_train_solution_similarity_gpt-3.5-instruct_final.npy

python analyze_discrete_similarity_train_gen_final.py
-> algorithms_train_solution_similarity_gen_codeonly_gpt-3.5-instruct_final.npy
python analyze_discrete_similarity_train_gen_final.py --field data_structures
-> data_structures_train_solution_similarity_gen_codeonly_gpt-3.5-instruct_final.npy


# [0327] parse GPT-4 annotated alg & ds
python analyze_discrete_similarity_train_gen_final.py \
    --model gpt-4 --field algorithms
-> algorithms_train_solution_similarity_gen_codeonly_gpt-4_final.npy

python analyze_discrete_similarity_train_gen_final.py \
    --model gpt-4 --field data_structures
-> data_structures_train_solution_similarity_gen_codeonly_gpt-4_final.npy

# [0328] for train src [GPT-4 annotations]
python analyze_discrete_similarity_train_src_final.py --field algorithms --model gpt-4
python analyze_discrete_similarity_train_src_final.py --field data_structures --model gpt-4

-> algorithms_train_solution_similarity_gpt-4_final.npy
-> data_structures_train_solution_similarity_gpt-4_final.npy
------------------------------------------------------------------

# generate alg ds via gpt-4

# [done]
python generation_alg_ds_traincode_src_final_gpt-4.py
python generation_alg_ds_traincode_gen_final_gpt-4.py

# TODO: 1. perform analysis based on gpt-4 generated summaries
# 2. compare with gpt-3.5-instruct generated comments