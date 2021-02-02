# Code for Identifying Implicit Discourse Relations with Linguistic-Feature-Enriched Neural Network

### 1) Download the PDTB data.

Follow the instructions [here](https://github.com/cgpotts/pdtb2). 

### 2) Download the brown cluster data.

Follow the instructions [here](https://www.cs.brandeis.edu/~clp/conll15st/dataset.html).

### 3) Download the pre-trained RoBERTa.large model.

Model | Description | # Params | Download
---|---|---|---
`roberta.large` | RoBERTa using the BERT-large architecture | 355M | [roberta.large.tar.gz](https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz)

### 4) Data preprocessing.

#### Input data of PDTB:
`train/dev/test.input0`: text data of disource argument 1 (one sample per line) 

`train/dev/test.input1`: text data of disource argument 2 (one sample per line)

`train/dev/test.label`:  labels (one label per line)

#### Preprocessing:
```bash
cd utils4data
bash pipeline_for_preprocess_data.sh
cd ../
bash preprocess.sh
```

### 5) Training on the PDTB dataset.

```bash
TOTAL_NUM_UPDATES=12000             # 10 epochs through IMDB for bsz 32.
WARMUP_UPDATES=400                  # 6 percent of the number of updates.
LR=5e-06                            # Peak LR for polynomial LR scheduler.
HEAD_NAME=pdtb_head                 # Custom name for the classification head.
NUM_CLASSES=11                      # Number of classes for the classification task.
MAX_SENTENCES=10                    # Batch size.
ROBERTA_PATH=roberta.large/model.pt # Path the the pre-trained RoBERTa checkpoint.
NUM_SEG=2                           # 0 for without segment embeddings.
GPU_ID=$1                           # gpu id. 
NUM_TYPE=$2                         # [1000, 3200] for number of Brown clusters.
SEED=$3                             # Random seed.

CUDA_VISIBLE_DEVICES=${GPU_ID} fairseq-train data-bin-${NUM_TYPE}/ \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --batch-size $MAX_SENTENCES \
    --max-tokens 2200 \
    --task sentence_prediction_brown \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large_brown \
    --user-dir roberta_model \
    --criterion sentence_prediction \
    --classification-head-name $HEAD_NAME \
    --num-classes $NUM_CLASSES \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.98)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --max-epoch 10 \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --shorten-method "truncate" \
    --find-unused-parameters \
    --seed ${SEED} \
    --num-segments ${NUM_SEG} \
    --num-type ${NUM_TYPE} \
    --update-freq 4 \
    --save-dir checkpoints.seg${NUM_SEG}.type${NUM_TYPE}.seed${SEED} \
    > train.log.seg${NUM_SEG}.type${NUM_TYPE}.seed${SEED} 2>&1 &
```


### 6) Inference on the PDTB test set.

```bash
    GPU_ID=$1
    NUM_TYPE=$2
    SEED=$3
    python predict.py ${GPU_ID} ${NUM_TYPE} ${SEED}
```
