import numpy as np
import argparse
import pickle
import json

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='personalized')
parser.add_argument('--model', type=str, default='llama-2-13b')
args = parser.parse_args()

infile = f'perplexity_scores/perplexity_goodreads_completions_{args.mode}_{args.model}-chat_500.npy'
ppl_scores = np.load(infile)

book_maps = json.load(open('../review_data/goodreads/book_maps_id_title.json'))
titles = list(book_maps.keys())

print(ppl_scores.shape, len(titles))

T_list = [0.5, 0.8, 1.0, 1.2, 1.5]
P_list = [0.90, 0.95, 0.98, 1.00]
indices_dict = {
    T: 
        {P: 
            {title: None for title in titles}
            for P in P_list}
        for T in T_list}
print(len(indices_dict[0.5][0.90]))

for i in range(len(T_list)):
    for j in range(len(P_list)):
        for k in range(ppl_scores.shape[-1]):
            ind_20_25 = np.where(np.logical_and(ppl_scores[i,j,:,k]>20, ppl_scores[i,j,:,k]<=25))[0]
            ind_25_30 = np.where(np.logical_and(ppl_scores[i,j,:,k]>25, ppl_scores[i,j,:,k]<=30))[0]
            ind_30_35 = np.where(np.logical_and(ppl_scores[i,j,:,k]>30, ppl_scores[i,j,:,k]<=35))[0]
            ind_35_40 = np.where(np.logical_and(ppl_scores[i,j,:,k]>35, ppl_scores[i,j,:,k]<=40))[0]
            ind_40_45 = np.where(np.logical_and(ppl_scores[i,j,:,k]>40, ppl_scores[i,j,:,k]<=45))[0]
            ind_45_50 = np.where(np.logical_and(ppl_scores[i,j,:,k]>45, ppl_scores[i,j,:,k]<=50))[0]
            indices_dict[T_list[i]][P_list[j]][titles[k]] = {
                '20-25': ind_20_25,
                '25-30': ind_25_30,
                '30-35': ind_30_35,
                '35-40': ind_35_40,
                '40-45': ind_40_45,
                '45-50': ind_45_50
            }
        
out_path = f'indices_dict_segment_goodreads_{args.mode}_{args.model}-chat_500.pkl'
print(out_path)
pickle.dump(indices_dict, open(out_path, 'wb'))