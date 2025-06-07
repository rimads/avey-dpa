export NUMBER_OF_GPUS=1
export BATCH_SIZE=2

torchrun --nproc_per_node=$NUMBER_OF_GPUS --nnodes=1 -m $MODEL_NAME.train --device_bsz $BATCH_SIZE
