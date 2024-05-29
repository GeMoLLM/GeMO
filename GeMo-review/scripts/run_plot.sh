---------------------------------------------
python plot_sentiment.py

---------------------------------------------
## mean scores
# compare within one model

python plot_sentiment_stacked_bar.py --model llama-2-13b --mode personalized
python plot_sentiment_stacked_bar.py --model llama-2-13b --mode personation
python plot_sentiment_stacked_bar.py --model vicuna-13b --mode personalized
python plot_sentiment_stacked_bar.py --model vicuna-13b --mode personation
python plot_sentiment_stacked_bar.py --model gpt-3.5-instruct --mode personalized
python plot_sentiment_stacked_bar.py --model gpt-3.5-instruct --mode personation

python plot_sentiment_stacked_bar_decay.py --model llama-2-13b --mode personalized;\
python plot_sentiment_stacked_bar_decay.py --model llama-2-13b --mode personation;\
python plot_sentiment_stacked_bar_decay.py --model vicuna-13b --mode personalized;\
python plot_sentiment_stacked_bar_decay.py --model vicuna-13b --mode personation;\

python plot_sentiment_stacked_bar_decay.py --model llama-2-13b --mode personalized -et 1.2 -p 50;\ 
python plot_sentiment_stacked_bar_decay.py --model llama-2-13b --mode personation -et 1.2 -p 50;\
python plot_sentiment_stacked_bar_decay.py --model vicuna-13b --mode personalized -et 1.2 -p 50;\
python plot_sentiment_stacked_bar_decay.py --model vicuna-13b --mode personation -et 1.2 -p 50;\


python plot_sentiment_stacked_bar_demonstration.py --model llama-2-13b --mode personalized

# compare personalized vs. personation
python plot_sentiment_stacked_bar_mode_cmp.py
python plot_sentiment_stacked_bar_mode_cmp.py --model vicuna-13b

# compare llama vs. vicuna
python plot_sentiment_stacked_bar_model_cmp.py
python plot_sentiment_stacked_bar_model_cmp.py --mode personation

# compare llama-chat, vicuna, llama-nonchat
python plot_sentiment_stacked_bar_all_models_cmp.py

# compare chat and non-chat
python plot_sentiment_stacked_bar_chat_cmp.py
python plot_sentiment_stacked_bar_chat_cmp.py --mode personation


# overall - compare T=1.0 vs. lin
python plot_sentiment_stacked_bar_decay_overall_cmp.py
python plot_sentiment_stacked_bar_decay_overall_cmp.py --model vicuna-13b

python plot_sentiment_stacked_bar_decay_config_overall_cmp.py -et 1.2 -p 50
python plot_sentiment_stacked_bar_decay_config_overall_cmp.py --model vicuna-13b -et 1.2 -p 50


# T=1.5, P=[0.90, 0.95, 0.98, 1.00]
python plot_sentiment_stacked_bar_T-1.5.py --model llama-2-13b --mode personalized
python plot_sentiment_stacked_bar_T-1.5.py --model llama-2-13b --mode personation
python plot_sentiment_stacked_bar_T-1.5.py --model vicuna-13b --mode personalized
python plot_sentiment_stacked_bar_T-1.5.py --model vicuna-13b --mode personation

python plot_sentiment_stacked_bar_p.py --model llama-2-13b --mode personalized
python plot_sentiment_stacked_bar_p.py --model llama-2-13b --mode personation
python plot_sentiment_stacked_bar_p.py --model vicuna-13b --mode personalized
python plot_sentiment_stacked_bar_p.py --model vicuna-13b --mode personation

# gpt-3.5-instruct vs gpt-4
python plot_sentiment_stacked_bar_gpt_cmp.py

---------------------------------------------
## entropy scores
python plot_sentiment_entropy.py

# compare within one model
python plot_sentiment_entropy_histplot_mode_cmp.py

python plot_sentiment_entropy_histplot_model_cmp.py

---------------------------------------------
python plot_topic_entropy.py
python plot_topic_entropy.py --model vicuna-13b --mode personation
python plot_topic_entropy.py --model llama-2-13b --mode personation

python plot_topic_entropy.py --model gpt-3.5-instruct --mode personalized
python plot_topic_entropy.py --model gpt-3.5-instruct --mode personation


python plot_topic_entropy_kdeplot_mode_cmp.py
python plot_topic_entropy_kdeplot_model_cmp.py

# compare chat and non-chat
python plot_topic_entropy_kdeplot_chat_cmp.py --mode personalized
python plot_topic_entropy_kdeplot_chat_cmp.py --mode personation


python plot_topic_entropy_kdeplot_decay_overall_cmp.py --model llama-2-13b -et 1.2 -p 50
python plot_topic_entropy_kdeplot_decay_overall_cmp.py --model vicuna-13b -et 1.2 -p 50

# re-plot 1st fig
python plot_topic_entropy_kdeplot_decay_overall_cmp_1.py -et 1.2 -p 50

python plot_legend_topic_entropy.py
python plot_legend_topic_entropy_more.py

# T=1.5, P=[0.90, 0.95, 0.98, 1.00]
python plot_topic_entropy_T-1.5.py
python plot_topic_entropy_T-1.5.py --model vicuna-13b --mode personalized
python plot_topic_entropy_T-1.5.py --model vicuna-13b --mode personation
python plot_topic_entropy_T-1.5.py --model llama-2-13b --mode personation

# gpt-3.5-instruct vs gpt-4
python plot_topic_entropy_kdeplot_gpt_cmp.py

---------------------------------------------

python plot_topic_distr_barplot.py
python plot_topic_distr_barplot_model_cmp.py
python plot_topic_distr_barplot_mode_cmp.py

# compare chat and non-chat
python plot_topic_distr_barplot_chat_cmp.py --mode personalized
python plot_topic_distr_barplot_chat_cmp.py --mode personation

python plot_topic_distr_barplot.py --model gpt-3.5-instruct --mode personalized
python plot_topic_distr_barplot.py --model gpt-3.5-instruct --mode personation

# gpt-3.5-instruct vs gpt-4
python plot_topic_distr_barplot_gpt_cmp.py


python plot_legend_gpt_cmp.py

# re-plot 1st fig
python plot_topic_distr_barplot_1.py

---------------------------------------------

python plot_wordfreq_simmat.py --model llama-2-13b --mode personalized --front
python plot_wordfreq_simmat.py --model llama-2-13b --mode personation --front
python plot_wordfreq_simmat.py --model vicuna-13b --mode personalized --front
python plot_wordfreq_simmat.py --model vicuna-13b --mode personation
python plot_wordfreq_simmat.py --model gpt-3.5-instruct --mode personalized
python plot_wordfreq_simmat.py --model gpt-3.5-instruct --mode personation


python plot_wordfreq_entropy_heatmat.py --model llama-2-13b --mode personalized --front
python plot_wordfreq_entropy_heatmat.py --model llama-2-13b --mode personation --front
python plot_wordfreq_entropy_heatmat.py --model vicuna-13b --mode personalized --front
python plot_wordfreq_entropy_heatmat.py --model vicuna-13b --mode personation
python plot_wordfreq_entropy_heatmat.py --model gpt-3.5-instruct --mode personalized
python plot_wordfreq_entropy_heatmat.py --model gpt-3.5-instruct --mode personation

