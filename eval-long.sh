python3 -m $MODEL_NAME.eval --model $MODEL_NAME --model_args pretrained=$MODEL_PATH --tasks niah_single_1,niah_single_2 --metadata='{"max_seq_lengths":[2048, 8192, 16384, 65535]}'
