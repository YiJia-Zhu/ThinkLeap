#!/bin/bash

# set -e: Exit immediately if a command exits with a non-zero status.
# set -x: Print commands and their arguments as they are executed. (Uncomment for debugging)
set -e
# set -x

# export DATASET="math500"
# export MODEL_NAME="Qwen3-14B"
# export CUDA_VISIBLE_DEVICES="1"

export WINDOW_SIZE=10
export PEAK_PROMINENCE=0.1
export RISE_MAGNITUDE=0.01
export EPOCH_NUM="10"
EPOCHLIST=("1" "3" "5" "10")

# if [[ "$MODEL_NAME" == *"32B"* ]]; then
#     export DEEPSPEED_3="--deepspeed cache/ds_z3_config.json"
# elif [[ "$MODEL_NAME" == *"14B"* && "$DATASET" == *"math"* ]]; then
#     export DEEPSPEED_3="--deepspeed cache/ds_z3_config.json"
# elif [[ "$MODEL_NAME" == *"7B"* && "$DATASET" == *"arc"* ]]; then
#     export DEEPSPEED_3="--deepspeed cache/ds_z3_config.json"
# else
#     export DEEPSPEED_3=""
# fi
# export DEEPSPEED_3="--deepspeed cache/ds_z3_config.json"

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


echo "Starting main_exp search processing pipeline..."
echo "------------------------------------"
echo "DATASET:                $DATASET"
echo "MODEL_NAME:          $MODEL_NAME"
echo "CUDA_VISIBLE_DEVICES:$CUDA_VISIBLE_DEVICES"
echo "------------------------------------"

# echo ""
# echo ">>> Step 1: Genrating confidence...20min"
# python step1_inference_and_calculate_confidence.py --model_path "${BASE_PATH}/huggingface_models/${MODEL_NAME}" --dataset_name ${DATASET} --window_size ${WINDOW_SIZE} --peak_prominence ${PEAK_PROMINENCE} --rise_magnitude ${RISE_MAGNITUDE}  --cuda_visible_devices ${CUDA_VISIBLE_DEVICES}

# echo ""
# echo ">>> Step 2: Running initial data processing...20min"
# python step2_replace_and_prune.py --model_path "${BASE_PATH}/huggingface_models/${MODEL_NAME}" --dataset_name ${DATASET} --window_size ${WINDOW_SIZE} --peak_prominence ${PEAK_PROMINENCE} --rise_magnitude ${RISE_MAGNITUDE}  --cuda_visible_devices ${CUDA_VISIBLE_DEVICES} --output_jsonl_file "${BASE_PATH}/step2_compressed_data/main_exp/${MODEL_NAME}/${FILENAME_BASE}.jsonl" 


# echo ""
# echo ">>> Step 3: Converting data for LLM factory..."
# python step3_data2LLMfactory.py --input_file "${BASE_PATH}/step2_compressed_data/main_exp/${MODEL_NAME}/${FILENAME_BASE}.jsonl" --output_file "${BASE_PATH}/step3_training_data/main_exp/${MODEL_NAME}/${FILENAME_BASE}.json" --test_split_ratio 0.0


# echo ""
# echo ">>> Step 4: Training using LLM factory...15min"
# llamafactory-cli train \
#     --stage sft \
#     --do_train True \
#     --model_name_or_path ${BASE_PATH}/huggingface_models/${MODEL_NAME} \
#     --preprocessing_num_workers 16 \
#     --finetuning_type lora \
#     --template ${MODEL_TEMPLATE} \
#     --flash_attn auto \
#     --dataset_dir ./step3_training_data/main_exp/${MODEL_NAME} \
#     --dataset ${FILENAME_BASE} \
#     --cutoff_len 14336 \
#     --learning_rate 5e-05 \
#     --num_train_epochs ${EPOCH_NUM} \
#     --max_samples 100000 \
#     --per_device_train_batch_size 2 \
#     --gradient_accumulation_steps 8 \
#     --lr_scheduler_type cosine \
#     --max_grad_norm 1.0 \
#     --logging_steps 5 \
#     --save_strategy epoch \
#     --warmup_steps 10 \
#     --packing False \
#     --enable_thinking True \
#     --report_to none \
#     --output_dir saves/${MODEL_NAME}/main_exp/${FILENAME_BASE} \
#     --bf16 True \
#     --plot_loss True \
#     --trust_remote_code True \
#     --ddp_timeout 180000000 \
#     --include_num_input_tokens_seen True \
#     --optim adamw_torch \
#     --lora_rank 8 \
#     --lora_alpha 16 \
#     --lora_dropout 0 \
#     --lora_target all \
#     --val_size 0.1 \
#     --eval_strategy steps \
#     --eval_steps 100 \
#     --per_device_eval_batch_size 2 \
#     --overwrite_output_dir True \
#     --seed 1 ${DEEPSPEED_3}

# --- Step 4.2: 查找、排序并索引所有可用的 Checkpoint ---

CHECKPOINT_DIR_BASE="./saves/${MODEL_NAME}/main_exp/${FILENAME_BASE}"

# 使用 `readarray` (或 `mapfile`) 将排序后的 checkpoint 路径读入一个 bash 数组
# 数组是 0-indexed, 所以第 1 个 epoch 对应索引 0，第 N 个 epoch 对应索引 N-1
readarray -t SORTED_CHECKPOINTS < <(ls -d ${CHECKPOINT_DIR_BASE}/checkpoint-*/ | sort -V)

# 检查是否找到了任何 checkpoint
if [ ${#SORTED_CHECKPOINTS[@]} -eq 0 ]; then
    echo "Error: No checkpoint directories found in ${CHECKPOINT_DIR_BASE}"
    exit 1
fi

TOTAL_AVAILABLE_EPOCHS=${#SORTED_CHECKPOINTS[@]}
echo "Found a total of ${TOTAL_AVAILABLE_EPOCHS} available epochs."
echo ""

# --- Step 4.3: 遍历指定的 Epoch 列表并进行评估 ---

echo ">>> Starting evaluation for the selected epochs..."

for epoch_num in "${EPOCHLIST[@]}"; do
    # 检查请求的 epoch 是否在有效范围内
    if [ "${epoch_num}" -le 0 ] || [ "${epoch_num}" -gt "${TOTAL_AVAILABLE_EPOCHS}" ]; then
        echo "--- WARNING: Requested epoch ${epoch_num} is not available. Total epochs found: ${TOTAL_AVAILABLE_EPOCHS}. Skipping. ---"
        continue
    fi

    # 计算数组索引 (Epoch 1 -> Index 0)
    array_index=$((epoch_num - 1))

    # 从数组中获取对应的 checkpoint 路径
    lora_path=${SORTED_CHECKPOINTS[array_index]}

    echo "--- Evaluating Epoch ${epoch_num} (Checkpoint path: ${lora_path}) ---"

    # 使用 epoch_num 命名输出文件
    OUTPUT_FILENAME="${FILENAME_BASE}_epoch${epoch_num}.json"

    python step5_eval_vllm.py \
        --model_path "${BASE_PATH}/huggingface_models/${MODEL_NAME}" \
        --dataset_name ${DATASET} \
        --cuda_visible_devices ${CUDA_VISIBLE_DEVICES} \
        --lora_path "${lora_path}" \
        --output_file "./step5_eval_data/main_exp/${MODEL_NAME}/${OUTPUT_FILENAME}"
done



echo ""
echo ">>> Step 5: eval org model ...20min"
python step5_eval_vllm.py --model_path "${BASE_PATH}/huggingface_models/${MODEL_NAME}" --dataset_name ${DATASET} --cuda_visible_devices ${CUDA_VISIBLE_DEVICES} --output_file "./step5_eval_data/main_exp/${MODEL_NAME}/${DATASET}_org.json"


echo ""
echo "Pipeline finished successfully."
