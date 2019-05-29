#!/usr/bin/env bash
export PYTHONPATH=/home/rowanz/code/swag2

MAX_SEQ_LENGTH=128

MODE=$1

BATCH_SIZE=64
EPOCHS=10
LR=2e-5

# You will need to specify your google cloud output bucket here.
OUTPUT_DIR=gs://MY_OUTPUT_BUCKET/

python bert/run_bert.py \
  --output_dir=${OUTPUT_DIR} \
  --do_lower_case=True \
  --max_seq_length=${MAX_SEQ_LENGTH} \
  --do_train=True \
  --predict_val=True \
  --predict_test=True \
  --train_batch_size=${BATCH_SIZE} \
  --predict_batch_size=512 \
  --learning_rate=${LR} \
  --num_train_epochs=${EPOCHS} \
  --warmup_proportion=0.2 \
  --iterations_per_loop=1000 \
  --use_tpu=True \
  --tpu_name=$(hostname) \
  --bert_large=True  \
  --endingonly=False