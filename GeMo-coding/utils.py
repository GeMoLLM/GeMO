from itertools import combinations
import numpy as np
import re

def jaccard_index(set1, set2):
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union != 0 else 1

def jaccard_similarity_index(list_of_sets):
    if len(list_of_sets) < 2:
        return 0  # Cannot calculate similarity for less than 2 sets
    pairwise_jaccard_indices = [jaccard_index(set1, set2) for set1, set2 in combinations(list_of_sets, 2)]
    average_jaccard_index = sum(pairwise_jaccard_indices) / len(pairwise_jaccard_indices)
    return average_jaccard_index

def entropy(l):
    unique, counts = np.unique(l, return_counts=True)
    probs = counts / len(l)
    n_classes = np.count_nonzero(probs)
    if n_classes <= 1:
        return 0
    ent = 0.
    # Compute entropy
    for i in probs:
        ent -= i * np.log2(i)
    return ent

def normalize_big_o_notation(x):
    tmp = x.replace('max', '').replace('min', '').replace('log', '').replace('sqrt', '').replace('len', '').\
        replace('o(', '').replace('(', '').replace(')', '').replace('width', 'w').replace('height', 'h').replace('a1', 'a').replace('a2', 'b').\
            replace('n1', 'a').replace('n2', 'b').replace('m1', 'a').replace('m2', 'b').\
                replace('rows', 'a').replace('cols', 'b').replace('moves', 'a')
    ch = sorted(list(set(np.unique(list(tmp))).intersection(set([chr(x) for x in range(ord('a'), ord('z')+1)]))))
    d = {}
    if len(ch) == 1:
        d = {ch[0]: 'n'}
    elif len(ch) == 2:
        d = {ch[0]: 'n', ch[1]: 'm'}
    elif len(ch) == 3:
        d = {ch[0]: 'n', ch[1]: 'm', ch[2]: 'q'}
    elif len(ch) == 4:
        d = {ch[0]: 'n', ch[1]: 'm', ch[2]: 'q', ch[3]: 'r'}
    return x.replace('max', 'MAX').replace('min', 'MIN').replace('log', 'LOG').replace('sqrt', 'SQRT').replace('len', 'LEN').replace('o(', 'O(').replace('width', 'w').replace('height', 'h').\
        replace('a1', 'a').replace('a2', 'b').replace('n1', 'a').replace('n2', 'b').replace('m1', 'a').replace('m2', 'b').replace('rows', 'a').replace('cols', 'b').replace('moves', 'a').translate(str.maketrans(d))\
        .replace('MAX', 'max').replace('MIN', 'min').replace('LOG', 'log').replace('SQRT', 'sqrt').replace('LEN', 'len')

def extract_complexity(description):
    # Regular expression to match time complexity notation (e.g., O(n), O(n^2), O(log n), etc.)
    # print(description)
    pattern = r'O\((?:[^()]+|\([^()]*\))*\)'
    match = re.search(pattern, description)
    if match:
        return transform_big_o_notation(
            normalize_big_o_notation(match.group(0).lower().replace(' ', '').replace('*', '').replace('\\cdot', '').replace('\\times', '').replace('√', 'sqrt')))
    else:
        return -1

def transform_big_o_notation(x):
    x = x.replace('\\', '')
    if x == '':
        return -1
    if x in ['O(msqrt(n))', 'O(nsqrt(n))', 'O((n-m)sqrt(n))', 'O(nsqrtn)', 'O((n-m)sqrt(m))', 'O((50-n)sqrt(m))']:
        return 6
    if x == 'O(mnlog(m))' or x in ['O(n^2logn)', 'O(qmlog(m)+qn)', 'O(mnlog(n))', 'O(n^2logn+nlogn)']:
        return 8
    if x in ['O(1)', 'O(10^6)', 'O(3120)', 'O(3220)', 'O(100000)', 'O(100001)', 'O(200,000)', 'O(3121)']:
        return 1
    if x.startswith('O(log'):
        return 2
    if x.startswith('O(sqrt') or x == 'O(n^0.5)':
        return 3
    if x.startswith('O(nlog') or x.startswith('O(mlog') or x.startswith('O(n\log'):
        return 5
    if x in ['O(mn)', 'O(n^2)', 'O(nm)', 'O(mn/2)', 'O(mn/7)', 'O(mn/3)', 'O(nm/2)', 'O(nm+nq)', 'O(n(m+q))', 'O(n(m//7))', 'O(n(m/3))', 'O((n/3)(n/7))', 'O(nm/3)', 'O((nm)/21)', 'O(n^2+nlogn)', 'O(nlen(m))', 'O(qm+n)', 'O(m^2+nlogn)', 'O((m+2)(n+2))', 'O(rowscols)', 'O(qm+nq)']:
        return 7
    if 'nmq' in x or 'mqn' in x or 'qmn' in x or 'qnm' in x or 'mnq' in x or x in ['O(n^3)', 'O(nm/3m/7)', 'O(n(m/3)(m/7))', 'O(nm^2)', 'O(q(n+1)m)', 'O(mn^2)'] or 'nklen' in x:
        return 9
    if x in ['O(n^4)', 'O(m^2n^2)']:
        return 10
    if x == 'O(2^n)':
        return 11
    if x in ['O(n)', 'O(n1)', 'O(n/2)', 'O(n/7)', 'O(m/n)', 'O(n/3)', 'O(n/m)', 'O(n3415)', 'O(len(n))', 'O(10(n-m))', 'O(n534)', 'O(q/n+m/n)', 'O(moves1)'] \
        or x.startswith('O(m+') \
        or x.startswith('O(m-') \
        or x.startswith('O(n+') \
        or x.startswith('O(n-') \
        or x.startswith('O(q+') \
        or x.startswith('O(r+') \
        or x.startswith('O(max(') \
        or x.startswith('O(min('):
        return 4
    raise NotImplementedError(f'Unknown complexity: {x}')

def regularize_algorithms(data):
    ret_data = []
    for x in data:
        x = x.replace(',', '')
        if 'greedy' in x.lower():
            x = 'Greedy'
        elif 'regular expression' in x.lower():
            x = 'Regular Expression'
        elif 'and conquer' in x.lower():
            x = 'Divide and Conquer'
        elif 'division' in x.lower():
            x = 'division'
        elif 'prefix sum' in x.lower():
            x = 'prefix sum'
        elif 'modulo' in x.lower() or 'modular' in x.lower() or 'modulus' in x.lower():
            x = 'Modular Arithmetic'
        elif 'gcd' in x.lower():
            x = 'GCD'
        elif 'math' in x.lower():
            x = 'Math'
        elif 'other' in x.lower():
            x = 'Other'
        elif 'arithmetic operations' in x.lower():
            x = 'arithmetic'
        elif 'conditional' in x.lower():
            x = 'conditional'
        x = x.lower()
        ret_data.append(x)
    return ret_data

def regularize_ds(data):
    ret_data = []
    for x in data:
        x = x.lower()
        if 'tuple' in x:
            x = 'tuple'
        elif 'stack' in x:
            x = 'stack'
        elif 'n/a' in x:
            x = 'none'
        elif 'math' in x:
            x = 'math'
        elif 'counter' in x:
            x = 'counter'
        elif x == 'lists':
            x = 'list'
        elif 'none' in x:
            x = 'none'
        elif 'integer' in x:
            x = 'int'
        elif 'maps' in x:
            x = 'dictionaries'
        elif x == 'dictioanry':
            x = 'dictionaries'
        elif x == 'conditionals':
            x = 'conditional statements'
        elif 'hash table' in x:
            x = 'dictionaries'
        elif 'dictionar' in x:
            x = 'dictionaries'
        ret_data.append(x)
    return ret_data

def regularize_tags(data):
    ret_data = []
    for x in data:
        x = x.lower().replace('.', '')
        if 'constructive algorithms' in x:
            x = 'constructive algorithms'
        elif 'cumulative' in x:
            x = 'cumulative'
        elif 'sorting' in x:
            x = 'sorting'
        elif x == 'dp':
            x = 'dynamic programming'
        elif x == 'conditionals' or x == 'conditions':
            x = 'conditional statements'
        elif 'count' in x:
            x = 'counter'
        elif 'heap' in x:
            x = 'heap'
        elif 'queue' in x:
            x = 'queue'
        elif x == 'recursive':
            x = 'recursion'
        elif x == 'set':
            x = 'sets'
        elif x == 'probability':
            x = 'probabilities'
        elif 'prefix sum' in x:
            x = 'prefix sums'
        elif 'efficient algorithm' in x:
            x = 'efficient algorithms'
        ret_data.append(x)
    return ret_data