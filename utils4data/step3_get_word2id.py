import sys
import os
from tqdm import tqdm
import json

words = []
for split in ['dev', 'test', 'train']:
    for cate in ['0', '1']:
        name = split + '.input' + cate
        txt_list = [line.strip() for line in open(name).readlines()]
        for txt in tqdm(txt_list, desc=name):
            cur_words = [w.strip() for w in txt.split() if w.strip() != '']
            words += cur_words
words = set(words)

for cate in ['1000', '3200']:

    f = open('word2id_' + cate, 'w')
    name = 'brown_cluster_' + cate + '.txt'
    lines = open(name).readlines()
    word2id = {}
    id_list = []
    for line in tqdm(lines, desc=name):
        segs = line.strip().split('\t')
        if len(segs) < 3:
            continue
        word2id[segs[1]] = segs[0]
        id_list.append(segs[0])
    id_list = list(set(id_list))
    id2id = {}
    for index, idx in enumerate(id_list):
        id2id[idx] = index + 3

    open('id2id_' + cate, 'w').write(json.dumps(id2id, ensure_ascii=False, indent=2))

    word2id_new = {}
    for word, idx in word2id.items():
        word2id_new[word] = id2id[idx]

    count = 0
    for word in words:
        if word in word2id_new:
            f.write(word + '\t' + str(word2id_new[word]) + '\n')
        else:
            f.write(word + '\t' + str(2) + '\n')
            count += 1
    print('oov: ', count)



