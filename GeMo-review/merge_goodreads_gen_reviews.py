import argparse
import json
import os
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument('--input_prefix', type=str, default='goodreads_completions_llama-2-13b-chat_500')
parser.add_argument('--output_folder', type=str, default='goodreads_llama-2-13b-chat')
parser.add_argument('--n_gen', type=int, default=10)
args = parser.parse_args()

gen_list = []
for i in range(args.n_gen):
    gen_list.append(json.load(open(f'{args.input_prefix}-{i}_merged.json')))

gen_all = []
for i in range(len(gen_list[0])):
    gen_all.append([gen_list[j][i] for j in range(args.n_gen)])

if not osp.exists(args.output_folder):
    os.makedirs(args.output_folder)

book_maps = json.load(open('../review_data/goodreads/book_maps_id_title.json'))

for book_id, gen_review in zip(book_maps.keys(), gen_all):
    # print(book_id, gen_review)
    out_path = osp.join(args.output_folder, book_id+'.jsonl')
    f = open(out_path, 'w')
    for rev in gen_review:
        f.write(json.dumps(rev)+'\n')
