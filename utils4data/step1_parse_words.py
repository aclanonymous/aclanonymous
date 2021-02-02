import regex as re
import os
from tqdm import tqdm

pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

for split in ['dev', 'test', 'train']:
    for cate in ['0', '1']:
        name = split + '.input' + cate
        lines = [line.strip() for line in open(name).readlines()]
        with open(name, 'w') as f:
            for line in tqdm(lines, desc=name):
                tokens = [token.strip() for token in re.findall(pat, line)]
                f.write(' '.join(tokens) + '\n')

