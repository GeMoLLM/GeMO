import argparse
import os.path as osp

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, default="personalized")
parser.add_argument("--model", type=str, default="llama-2-13b")
args = parser.parse_args()

cnt = 0
for temperature in [0.5, 0.8, 1.0, 1.2]:
    for top_p in ["0.90", "0.95", "0.98", "1.00"]:
        for i in range(10):
            filename = f'goodreads_completions_{args.mode}_{args.model}-chat_500_temp-{temperature}_p-{top_p}_k-50-{i}_merged.json'
            if not osp.exists(filename):
                print(filename)
                cnt += 1
            # assert osp.exists(filename), f"File {filename} does not exist"
print(cnt)