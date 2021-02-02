from tqdm import tqdm
import sys


for split in ['dev', 'test', 'train']:
    for cate in ['0', '1']:
        name = split + '.input' + cate + '.bpe'
        txt_list = [line.strip() for line in open(name).readlines()]
        num_list = [line.strip() for line in open(name + '.len').readlines()]
        for txt, num in tqdm(zip(txt_list, num_list), total=len(txt_list), desc=name):
            nums = [int(token.strip()) for token in num.strip().split() if token.strip() != '']
            assert len(txt.strip().split()) == sum(nums)


for split in ['dev', 'test', 'train']:
    for cate in ['0', '1']:
        name = split + '.input' + cate
        txt_list = [line.strip() for line in open(name).readlines()]
        num_list = [line.strip() for line in open(name + '.bpe.len').readlines()]
        for txt, num in tqdm(zip(txt_list, num_list), total=len(txt_list), desc=name):
            assert len(txt.split()) == len(num.split())

print('right.')
