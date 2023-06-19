#!/bin/bash
set -e

export WANDB_PROJECT='compare_eval'
export WANDB_ENTITY='griffinadams'

MODEL_SIZE=$1
EXPERIMENT=$2
export SUMMARIZATION_TASK=$3
VAL_NUM=512
MAX_MEMORY=49000
EVAL_STEPS=500
MAX_STEPS=10000
MAX_EVAL_SAMPLES=512

if [[ $MODEL_SIZE -eq 7 ]]; then
  echo "Detected 7B"
  SOURCE_MAX_LEN=2048
  TARGET_MAX_LEN=2048
  LEARNING_RATE=0.0002
else
  echo "Detected 40B"
  SOURCE_MAX_LEN=2048
  TARGET_MAX_LEN=2048
  LEARNING_RATE=0.0001
fi

python qlora.py \
  --dataset_format self-instruct \
  --dataset griffin/progressive_summarization \
  --run_name $EXPERIMENT \
  --max_eval_samples $MAX_EVAL_SAMPLES \
  --model_name_or_path "tiiuae/falcon-${MODEL_SIZE}b" \
  --learning_rate $LEARNING_RATE \
  --eval_dataset_size $VAL_NUM \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --source_max_len $SOURCE_MAX_LEN \
  --target_max_len $TARGET_MAX_LEN \
  --max_memory_MB $MAX_MEMORY \
  --report_to wandb \
  --save_total_limit 10 \
  --save_steps $EVAL_STEPS \
  --eval_steps $EVAL_STEPS \
  --evaluation_strategy steps \
  --trust_remote_code True \
  --max_steps $MAX_STEPS \
  --logging_steps 2 \
  --do_train \
  --do_eval \
  --bf16 \
  --output_dir $HOME/falcon_weights/$EXPERIMENT \
#  --train_on_source False \
