for SPLIT in train dev test; do
    for ID in 0 1; do
        echo $SPLIT $ID
        python -m multiprocessing_bpe_encoder_for_len \
            --encoder-json encoder.json \
            --vocab-bpe vocab.bpe \
            --inputs "$SPLIT.input$ID" \
            --outputs "$SPLIT.input$ID.bpe.len" \
            --workers 20 \
            --keep-empty
    done
done
