from tqdm import tqdm


for split in ['dev', 'test', 'train']:
    for cate in ['0', '1']:
        result = []
        for line in tqdm(open(split + '.input' + cate + '.bpe.len').readlines(), desc=cate):
            segs = [int(w.strip()) for w in line.strip().split()]
            if len(segs) > 3:
                cur = ['1'] * sum(segs[:3]) + ['0'] * sum(segs[3:-1]) + ['1'] * segs[-1]
            else:
                cur = ['1'] * sum(segs)
            result.append(' '.join(cur) + '\n')
            assert sum(segs) == len(cur)
        open(split + '.input' + cate + '.mask', 'w').writelines(result)
