import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[1]  # gpu id
from roberta_model.model import RobertaModel
from tqdm import tqdm
from collections import defaultdict

num_type = sys.argv[2]
seed = sys.argv[3]

def load_data():
    lines_0 = [line.strip() for line in open('utils4data/test.input0.bpe').readlines()]
    lines_1 = [line.strip() for line in open('utils4data/test.input1.bpe').readlines()]
    lines_2 = [line.strip() for line in open('utils4data/type_' + num_type + '/test.input2.type').readlines()]
    lines_3 = [line.strip() for line in open('utils4data/test.input3.segment').readlines()]
    lines_4 = [line.strip() for line in open('utils4data/test.input0.mask').readlines()]
    lines_5 = [line.strip() for line in open('utils4data/test.input1.mask').readlines()]
    labels = [line.strip() for line in open('utils4data/test.label').readlines()]

    assert len(lines_0) == len(lines_1) == len(lines_2) == len(lines_3) == len(labels) == len(lines_4) == len(lines_5)

    input0_list = []
    input1_list = []
    input2_list = []
    input3_list = []

    input4_list = []
    input5_list = []
    src2labels= defaultdict(list)
    for input0, input1, input2, input3, label, input4, input5 in tqdm(zip(lines_0, lines_1, lines_2, lines_3, labels, lines_4, lines_5), total=len(labels), desc='load data'):
        src = input0 + ' [SEP] ' + input1
        if src not in src2labels:
            input0_list.append(input0)
            input1_list.append(input1)
            input2_list.append(input2)
            input3_list.append(input3)

            input4_list.append(input4)
            input5_list.append(input5)

        src2labels[src].append(label)

    return input0_list, input1_list, input2_list, input3_list, src2labels, input4_list, input5_list



roberta = RobertaModel.from_pretrained('checkpoints.seg2.type' + num_type + '.seed' + seed, checkpoint_file='checkpoint_best.pt', data_name_or_path='data-bin-' + num_type)  # model path and dict path
roberta.eval()  # disable dropout
label_fn = lambda label: roberta.task.label_dictionary.string([label + roberta.task.label_dictionary.nspecial])

roberta = roberta.to('cuda')
print(roberta)

input0_list, input1_list, input2_list, input3_list, src2labels, input4_list, input5_list = load_data()

right = 0.
tot = 0.
for input0, input1, input2, input3, input4, input5 in tqdm(zip(input0_list, input1_list, input2_list, input3_list, input4_list, input5_list), total=len(input0_list), desc='testing'):
    tokens, type_tokens, segment_labels, mask0, mask1 = roberta.encode(input0, input1, input2, input3, input4, input5)
    pred = label_fn(roberta.predict('pdtb_head', tokens=tokens, type_tokens=type_tokens, segment_labels=segment_labels, mask0=mask0, mask1=mask1).argmax().item())
    tot += 1
    if pred in src2labels[input0 + ' [SEP] ' + input1]:
        right += 1

print('acc: ', right / tot)


