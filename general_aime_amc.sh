#!/bin/bash

# set -e: Exit immediately if a command exits with a non-zero status.
# set -x: Print commands and their arguments as they are executed. (Uncomment for debugging)
set -e
# set -x

# export DATASET="aime24"
# export MODEL_NAME="Qwen3-14B"
export LORA_DATASET="gsm8k"

# export CUDA_VISIBLE_DEVICES="2,3"

export WINDOW_SIZE=10
export PEAK_PROMINENCE=0.1
export RISE_MAGNITUDE=0.01
if [[ "$MODEL_NAME" == *"QwQ"* ]]; then
    export MODEL_TEMPLATE="qwen"
elif [[ "$MODEL_NAME" == *"Qwen3"* ]]; then
    export MODEL_TEMPLATE="qwen3"
elif [[ "$MODEL_NAME" == *"DeepSeek-R1"* ]]; then
    export MODEL_TEMPLATE="deepseekr1"
else
    export MODEL_TEMPLATE="qwen"
fi

BASE_PATH=$(pwd)

FILENAME_BASE="${DATASET}_${WINDOW_SIZE}_${PEAK_PROMINENCE}_${RISE_MAGNITUDE}"
LORA_BASE="${LORA_DATASET}_${WINDOW_SIZE}_${PEAK_PROMINENCE}_${RISE_MAGNITUDE}"


echo "Starting main_exp717 search processing pipeline..."
echo "------------------------------------"
echo "DATASET:                $DATASET"
echo "MODEL_NAME:          $MODEL_NAME"
echo "CUDA_VISIBLE_DEVICES:$CUDA_VISIBLE_DEVICES"
echo "------------------------------------"

echo ""
echo ">>> Step 4: eval model...20min"
python step4_eval_vllm.py --model_path "./huggingface_models/${MODEL_NAME}" --dataset_name ${DATASET} --cuda_visible_devices ${CUDA_VISIBLE_DEVICES} --lora_path "./saves/${MODEL_NAME}/main_exp717/${LORA_BASE}" --output_file "./step5_eval_data/main_exp717/${MODEL_NAME}/${FILENAME_BASE}.json"



echo ""
echo ">>> Step 5: eval org model ...20min"
python step4_eval_vllm.py --model_path "./huggingface_models/${MODEL_NAME}" --dataset_name ${DATASET} --cuda_visible_devices ${CUDA_VISIBLE_DEVICES} --output_file "./step5_eval_data/main_exp717/${MODEL_NAME}/${DATASET}_org.json"


echo ""
echo "Pipeline finished successfully."
