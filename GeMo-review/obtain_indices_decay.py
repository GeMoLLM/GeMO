import numpy as np
import argparse
import pickle
import json

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default='personalized')
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--end_temperature', '-et', type=float, default=1.0)
parser.add_argument('--period', '-p', type=int, default=20)
args = parser.parse_args()

infile = f'perplexity_scores/perplexity_goodreads_completions_decay_{args.mode}_{args.model}-chat_500.npy' \
    if args.end_temperature == 1.0 and args.period == 20 \
        else f'perplexity_scores/perplexity_goodreads_completions_decay-{args.end_temperature}-{args.period}_{args.mode}_{args.model}-chat_500.npy'
ppl_scores = np.load(infile)

book_maps = json.load(open('../review_data/goodreads/book_maps_id_title.json'))
titles = list(book_maps.keys())

print(ppl_scores.shape, len(titles))

decay_list = ['linear', 'exponential']
P_list = [0.90, 0.95, 0.98, 1.00]
indices_dict = {
    decay: 
        {P: 
            {title: None for title in titles}
            for P in P_list}
        for decay in decay_list}

for i in range(2):
    for j in range(4):
        for k in range(ppl_scores.shape[-1]):
            indices_dict[decay_list[i]][P_list[j]][titles[k]] = \
                np.where(ppl_scores[i,j,:,k]<=20)[0]
        
out_path = f'indices_dict_goodreads_decay_{args.mode}_{args.model}-chat_500.pkl' \
    if args.end_temperature == 1.0 and args.period == 20 \
        else f'indices_dict_goodreads_decay-{args.end_temperature}-{args.period}_{args.mode}_{args.model}-chat_500.pkl'
print(out_path)
pickle.dump(indices_dict, open(out_path, 'wb'))