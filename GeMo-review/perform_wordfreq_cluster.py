import argparse
import spacy
import os
import os.path as osp
from tqdm import tqdm
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from gensim.test.utils import common_texts
from gensim.models import Word2Vec
import gensim
import gensim.downloader
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='llama-2-13b')
parser.add_argument('--mode', type=str, default='personalized')
parser.add_argument('--device', '-d', type=int, default=0)
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.device)

folder = 'analysis_results_wordfreq'
gen_path = osp.join(folder, f'summary_wordfreq_goodreads_{args.mode}_{args.model}-chat_500.pkl')

gen_data = pickle.load(open(gen_path, 'rb'))

T_list = [0.5, 0.8, 1.0, 1.2]
P_list = [0.90, 0.95, 0.98, 1.00]

glove_vectors = gensim.downloader.load('glove-twitter-25')
print('load in glove vectors!')

out_emb_folder = 'wordfreq_embeddings/'
out_folder = 'results_wordfreq_clustering/'
os.makedirs(out_emb_folder, exist_ok=True)
os.makedirs(out_folder, exist_ok=True)

all_embs = {T: {P: [] for P in P_list} for T in T_list}
entropy = np.zeros((4, 4))
for ti, T in enumerate(T_list):
    for pi, P in enumerate(P_list):
        gen_counter = gen_data[T][P]
        embs = []
        for word in gen_counter.keys():
            if word in glove_vectors:
                embs.append(glove_vectors[word])
        all_embs[T][P] = np.array(embs)
np.save(f'{out_emb_folder}/wordfreq_embeddings_{args.mode}_{args.model}-chat_500.npy', all_embs)
print('extracting embeddings done!')

def cluster(X):
    if len(X) <= 2:
        return -1
    n_clusters = []
    sil_scores = []
    for n in tqdm(range(2, min(50, len(X)))):
        kmeans = KMeans(n_clusters=n, random_state=0, n_init="auto").fit(X)
        labels = kmeans.fit_predict(X)
        silhouette_avg = silhouette_score(X, labels)
        n_clusters.append(n)
        sil_scores.append(silhouette_avg)
    idx = np.argmax(sil_scores)
    print(sil_scores)
    return n_clusters[idx]

for T in T_list:
    for P in P_list:
        embs = all_embs[T][P]
        assert len(embs) > 0
        print(f'clustering for T={T}, P={P}')
        n_clusters = cluster(embs)
