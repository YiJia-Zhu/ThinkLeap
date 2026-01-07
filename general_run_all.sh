#!/bin/bash

# set -e: Exit immediately if a command exits with a non-zero status.
# 如果任何一次参数组合运行失败，整个脚本将停止。
# 如果你想让脚本在某次失败后继续尝试下一个组合，请注释掉下面这行。
set -e
export CUDA_VISIBLE_DEVICES="4,5"

# --- 定义要遍历的参数列表 ---\

# DATASETS=("math500")
# MODEL_NAMES=("QwQ-32B")
# MODEL_NAMES=("QwQ-32B")



DATASETS=("aime24" "amc23")

# MODEL_NAMES=("DeepSeek-R1-Distill-Qwen-1.5B")


# MODEL_NAMES=("QwQ-32B")


MODEL_NAMES=("DeepSeek-R1-Distill-Qwen-1.5B" "Qwen3-14B" "QwQ-32B")




echo "Starting grid search for parameters..."
echo "------------------------------------------------"



# --- 开始三层循环 ---
for ws in "${DATASETS[@]}"; do
  for pp in "${MODEL_NAMES[@]}"; do
      
    current_run=$((current_run + 1))
    # 导出环境变量，这样 run_pipeline.sh 就可以读取到它们
    export DATASET=${ws}
    export MODEL_NAME=${pp}

    echo ""
    echo "######################################################################"
    echo "### Starting Run $current_run  ### $DATASET $MODEL_NAME"
    echo "######################################################################"
    

    # 调用执行单次流程的脚本
    bash ./general_aime_amc.sh
    # bash ./main_exp_aime_amc.sh


  done
done



echo ""
echo "######################################################################"
echo "All $total_runs parameter combinations have been processed."
echo "Grid search finished."
echo "######################################################################"