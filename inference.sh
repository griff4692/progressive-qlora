#!/bin/bash
set -e

MODEL_SIZE=$1
EXPERIMENT=$2
export SUMMARIZATION_TASK=$3

MAX_MEMORY=48000
MAX_EVAL_SAMPLES=1
TARGET_MIN_LEN=20
TARGET_MAX_LEN=512

if [[ $MODEL_SIZE -eq 7 ]]; then
  echo "Detected 7B"
  SOURCE_MAX_LEN=2048
else
  echo "Detected 40B"
  SOURCE_MAX_LEN=2048
fi


python generate.py \
  --run_name $EXPERIMENT \
  --max_eval_samples $MAX_EVAL_SAMPLES \
  --model_name_or_path "tiiuae/falcon-${MODEL_SIZE}b" \
  --per_device_eval_batch_size 8 \
  --source_max_len $SOURCE_MAX_LEN \
  --min_new_tokens $TARGET_MIN_LEN \
  --max_new_tokens $TARGET_MAX_LEN \
  --max_memory_MB $MAX_MEMORY \
  --trust_remote_code True \
  --predict_with_generate \
  --num_beams 1 \
  --no_repeat_ngram_size 3 \
  --output_dir $HOME/falcon_weights/$EXPERIMENT \
#  --bf16 \
