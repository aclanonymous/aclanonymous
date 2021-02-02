#!/bin/bash

./bpe.sh
./bpe_for_len.sh
python -u step1_parse_words.py
python -u step2_check.py
python -u step3_get_word2id.py
python -u step4_make_brown_and_segment.py
python -u step5_get_gather_mask.py
