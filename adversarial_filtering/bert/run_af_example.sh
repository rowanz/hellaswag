#!/usr/bin/env bash

# Here's an example script for running AF. Replace MYBUCKETNAME with your output bucket, and you'll need to somehow get wikihow samples there.

numsents=3
num_train_epochs=3.0
num_negatives=5

python train_af.py \
  --train_batch_size=64 \
  --predict_batch_size=2048 \
  --max_seq_length=144 \
  --max_learning_rate=4e-5 \
  --min_learning_rate=1e-5 \
  --num_train_epochs=$num_train_epochs \
  --output_dir=gs://MYBUCKETNAME/af-${HOSTNAME}/ \
  --use_tpu=True \
  --tpu_name=${HOSTNAME} \
  --assignments_dir=gs://MYBUCKETNAME/af-wikihow-${numsents}sent-${HOSTNAME}-feb23/ \
  --assignments_fn=gs://MYBUCKETNAME/wikihow/samples.jsonl \
  --bert_large=True \
  --random_seed=123456 \
  --subsample=3 \
  --weight_af=True \
  --lm_loss=False \
  --warmup_proportion=0.2 \
  --save_checkpoints_steps=1000 \
  --iterations_per_loop=1000 \
  --num_negatives=${num_negatives} \
  --wikihow=True
