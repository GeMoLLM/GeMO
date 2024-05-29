python plot_pairwise_similarity_text.py -f description;\
python plot_pairwise_similarity_text.py -f functionality;\
python plot_pairwise_similarity_text.py -f algorithm;\
python plot_pairwise_similarity_text.py -f data_structure;\

python plot_pairwise_similarity_text_demonstration.py -f description --xmax 1.1

python plot_pairwise_similarity_text.py -m gpt-4 -f description;\
python plot_pairwise_similarity_text.py -m gpt-4 -f functionality;\
python plot_pairwise_similarity_text.py -m gpt-4 -f algorithm;\
python plot_pairwise_similarity_text.py -m gpt-4 -f data_structure;\

python plot_legend_text_cossim.py


------------------------------------------------------------
python plot_jaccard_index.py -f tags;\
python plot_jaccard_index.py -f algorithms;\
python plot_jaccard_index.py -f data_structures;\

python plot_jaccard_index_demonstration.py -f algorithms

------------------------------------------------------------
# complexity (population) distribution

python plot_complexity.py

------------------------------------------------------------
# complexity (sample) distribution

python plot_complexity_entropy.py

------------------------------------------------------------
# plot runtime
python plot_runtime.py

------------------------------------------------------------
python plot_acc.py

python plot_plagiarism_score.py

------------------------------------------------------------
# [0425] claude

python plot_pairwise_similarity_text.py -f description -c --xmax 1.2;\
python plot_pairwise_similarity_text.py -f functionality -c --xmax 1.2;\
python plot_pairwise_similarity_text.py -f algorithm -c --xmax 1.2;\
python plot_pairwise_similarity_text.py -f data_structure -c --xmax 1.2;\

python plot_jaccard_index.py -f tags -c;\
python plot_jaccard_index.py -f algorithms -c;\
python plot_jaccard_index.py -f data_structures -c;\

python plot_complexity.py -c
python plot_complexity_entropy.py -c

python plot_runtime.py -c
python plot_plagiarism_score.py -c --N 50