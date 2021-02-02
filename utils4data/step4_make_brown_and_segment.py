from tqdm import tqdm
import sys
import os

for tp in ['1000', '3200']:
    for split in ['dev', 'test', 'train']:
        brown_type_path = 'type_' + tp
        if not os.path.exists(brown_type_path):
            os.makedirs(brown_type_path)
        f = open(brown_type_path + '/' + split + '.input2.type', 'w')   # dev.input2.type
        f_v = open(brown_type_path + '/' + 'input2.vocab', 'w')   # input2.vocab
        v = [0, 1]

        f_seg = open(split + '.input3.segment', 'w')   # dev.input3.segment
        f_seg_v = open('input3.vocab', 'w')   # input3.vocab

        word2id = {}
        for line in open('word2id_' + tp).readlines():
            segs = line.strip().split('\t')
            word2id[segs[0]] = segs[1]
            
        name0 = split + '.input0'
        name1 = split + '.input1'
        txt0_list = [line.strip() for line in open(name0).readlines()]
        txt1_list = [line.strip() for line in open(name1).readlines()]
        num0_list = [line.strip() for line in open(name0 + '.bpe.len').readlines()]
        num1_list = [line.strip() for line in open(name1 + '.bpe.len').readlines()]

        for txt0, txt1, num0, num1 in tqdm(zip(txt0_list, txt1_list, num0_list, num1_list), total=len(txt0_list), desc=split):
            result_line = '0 '
            result_seg = '0 '
            assert len(txt0.split()) == len(num0.split())
            assert len(txt1.split()) == len(num1.split())
            
            tot_len = 4
            for w, n in zip(txt0.split(), num0.split()):
                for idx in range(int(n)):
                    result_line += word2id[w] + ' '
                    result_seg += '0' + ' '
                    tot_len += 1
                    v.append(int(word2id[w]))
            if tot_len > 4:
                result_line += '1 1' + ' '
                result_seg += '0 1' + ' '

            for w, n in zip(txt1.split(), num1.split()):
                for idx in range(int(n)):
                    result_line += word2id[w] + ' '
                    result_seg += '1' + ' '
                    tot_len += 1
                    v.append(int(word2id[w]))
            
            result_line += '1'
            result_seg += '1'
            assert(tot_len == len(result_line.split()))
            if result_line.endswith(' 1 1'):
                print(result_line)

            assert not result_line.endswith(' 1 1')
            assert len(result_line.split()) == len(result_seg.split())
            
            f.write(result_line + '\n')
            f_seg.write(result_seg + '\n')
        v = list(set(v))
        v = sorted(v)
        for val in v:
            f_v.write(str(val) + ' ' + str(val) + '\n')

        f_seg_v.write('0 0\n1 1\n')
