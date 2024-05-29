import numpy as np
import argparse
import pickle
import json

parser = argparse.ArgumentParser()
parser.add_argument('--T_list', type=str, default='0.5,0.8,1.0,1.2,1.5')
parser.add_argument('--P_list', type=str, default='0.90,0.95,0.98,1.00')
parser.add_argument('--mode', type=str, default='personalized')
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--chat', type=str, default='chat')
args = parser.parse_args()

infile = f'perplexity_scores/perplexity_goodreads_completions_{args.mode}_{args.model}-{args.chat}_500.npy'
ppl_scores = np.load(infile)

book_maps = json.load(open('../review_data/goodreads/book_maps_id_title.json'))
titles = list(book_maps.keys())

print(ppl_scores.shape, len(titles))

T_list = [float(x) for x in args.T_list.split(',')]
P_list = [float(p) for p in args.P_list.split(',')]

indices_dict = {
    T: 
        {P: 
            {title: None for title in titles}
            for P in P_list}
        for T in T_list}
print(len(indices_dict[T_list[0]][P_list[0]]))

for i in range(len(T_list)):
    for j in range(len(P_list)):
        for k in range(ppl_scores.shape[-1]):
            indices_dict[T_list[i]][P_list[j]][titles[k]] = \
                np.where(np.logical_and(ppl_scores[i,j,:,k]<=20, ppl_scores[i,j,:,k]>0))[0]
        
out_path = f'indices_dict_goodreads_{args.mode}_{args.model}-{args.chat}_500.pkl'
print(out_path)
pickle.dump(indices_dict, open(out_path, 'wb'))