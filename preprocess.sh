fairseq-preprocess \
    --only-source \
    --trainpref "utils4data/train.input0.bpe" \
    --validpref "utils4data/dev.input0.bpe" \
    --destdir "data-bin-1000/input0" \
    --workers 60 \
    --srcdict roberta.large/dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "utils4data/train.input1.bpe" \
    --validpref "utils4data/dev.input1.bpe" \
    --destdir "data-bin-1000/input1" \
    --workers 60 \
    --srcdict roberta.large/dict.txt

fairseq-preprocess \
    --only-source \
    --trainpref "utils4data/type_1000/train.input2.type" \
    --validpref "utils4data/type_1000/dev.input2.type" \
    --destdir "data-bin-1000/input2" \
    --workers 60 \
    --srcdict utils4data/type_1000/input2.vocab

fairseq-preprocess \
    --only-source \
    --trainpref "utils4data/train.input3.segment" \
    --validpref "utils4data/dev.input3.segment" \
    --destdir "data-bin-1000/input3" \
    --workers 60 \
    --srcdict utils4data/input3.vocab

fairseq-preprocess \
    --only-source \
    --trainpref "utils4data/train.input0.mask" \
    --validpref "utils4data/dev.input0.mask" \
    --destdir "data-bin-1000/input4" \
    --workers 60 \
    --srcdict utils4data/input3.vocab

fairseq-preprocess \
    --only-source \
    --trainpref "utils4data/train.input1.mask" \
    --validpref "utils4data/dev.input1.mask" \
    --destdir "data-bin-1000/input5" \
    --workers 60 \
    --srcdict utils4data/input3.vocab

fairseq-preprocess \
    --only-source \
    --trainpref "utils4data/train.label" \
    --validpref "utils4data/dev.label" \
    --destdir "data-bin-1000/label" \
    --workers 60
