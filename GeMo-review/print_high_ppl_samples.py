import json
import numpy as np
import pickle
import os.path as osp
import random

model = 'llama-2-13b'
mode = 'personalized'

titles = list(json.load(open('../SafeNLP/results_sentiment/sentiment_goodreads_src_grouped_reviews_long_sub_en_10_sentiment.json')).keys())
infile = f'perplexity_scores/perplexity_goodreads_completions_{mode}_{model}-chat_500.npy'
titles = set([title.replace('.jsonl', '') for title in titles])

ppl = np.load(infile)
print(ppl.shape)
print(len(titles))

indices = pickle.load(open(f'indices_dict_segment_goodreads_{mode}_{model}-chat_500.pkl', 'rb'))
indices = indices[1.5][1.0]
# print(indices.keys())
intervals = ['20-25', '25-30', '30-35', '35-40', '40-45', '45-50']
items = {
    interval: [] for interval in intervals
}
for k, v in indices.items():
    for interval in intervals:
        items[interval].extend([(k,x) for x in v[interval]])

for interval in intervals:
    print('interval', interval)
    l_sub = random.sample(items[interval], 10)

    folder = f'folder_goodreads_completions_{mode}_{model}-chat_500_temp-1.5_p-1.00_k-50/'
    for k, x in l_sub:
        file = osp.join(folder, f'{k}.jsonl')
        with open(file) as f:
            cnt = 0
            for line in f:
                if cnt == x:
                    print(json.loads(line))
                    break
                cnt += 1
        print('------------------------------------------')