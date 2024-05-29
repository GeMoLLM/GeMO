import numpy as np

def get_mean(l):
    return np.mean(l)

def get_std(l):
    return np.std(l)

def get_entropy(l):
    _, count = np.unique(l, return_counts=True)
    if len(count) == 1:
        return 0
    return -np.sum(count/len(l) * np.log2(count/len(l)))

def transform_sentiment(l):
    unique = np.unique(l)
    if unique[0] == 'POSITIVE' or unique[0] == 'NEGATIVE':
        return [1 if x == 'POSITIVE' else 0 for x in l]
    assert unique[0] == 0 or unique[0] == 1, f'l contains {unique}'
    return l
