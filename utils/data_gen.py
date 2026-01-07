import argparse
import json
import os
import sys
from tqdm import tqdm
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.dataset_loader import *
from typing import List, Dict, Any, Tuple, Set
from scipy.signal import find_peaks
# export CUDA_VISIBLE_DEVICES=0,2
from vllm import LLM, SamplingParams


# Initialize logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# commonsense_qa
def parse_arguments_for_example():
    """Parses command-line arguments for the example script."""
    parser = argparse.ArgumentParser(description="Run model generation and pruning for a single math sample, with plotting.")
    parser.add_argument(
        "--dataset_name", type=str, default = "gsm8k",
        help="Name of the dataset")
    parser.add_argument(
        "--model_path",
        type=str,
        default="./huggingface_models/DeepSeek-R1-Distill-Qwen-1.5B",
        # default="./merge_models/Qwen3-14B-gsm8k-25",
        help="Path to the Hugging Face model for generation and pruning."
    )
    parser.add_argument(
        "--output_jsonl_file",
        type=str,
        default="./step2_compressed_data/main_exp/tmp.jsonl",
        help="Comma-separated list of GPU IDs to use (e.g., '0,1'). This determines the tensor_parallel_size."
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="0",
        help="Comma-separated list of GPU IDs to use (e.g., '0,1'). This determines the tensor_parallel_size."
    )

    

    ####### param
    parser.add_argument(
        "--window_size", type=int, default=10,
        help="for prune"
    )
    parser.add_argument(
        "--peak_prominence", type=float, default=0.1,
        help="peak_prominence"
    )
    parser.add_argument(
        "--rise_magnitude", type=float, default=0.01,
        help="rise_magnitude"
    )

    ######### ablation
    parser.add_argument(
        "--only_early_stop", type=int, default=0,
        help="only_early_stop"
    )
    parser.add_argument(
        "--is_skeptical", type=int, default=1,
        help="is_skeptical"
    )
    parser.add_argument(
        "--early_stop_threshold", type=float, default=0.95,
        help="early_stop_threshold"
    )


    #########
    parser.add_argument(
        "--growth_factor", type=float, default=1.1,
        help="Factor for growth_factor probability check in pruning strategy."
    )
    parser.add_argument(
        "--top_k_factor", type=float, default=2,
        help="Factor for top_k probability check in pruning strategy."
    )
    parser.add_argument(
        "--top_k_growth_factor", type=float, default=0.95,
        help="Factor for tok_k stability probability check in pruning strategy."
    )
    parser.add_argument(
        "--tokenizer_default_max_length", type=int, default=8192,
        help="Default model_max_length to set for the tokenizer if not set or too large."
    )
    parser.add_argument(
        "--tokenizer_max_length_override_threshold", type=int, default=20480,
        help="Threshold above which tokenizer's model_max_length will be overridden."
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device for initial generation if not using vLLM for it (e.g., 'cuda', 'cpu')."
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=4096,
        help="Maximum new tokens for initial solution generation."
    )
    parser.add_argument(
        "--do_sample", type=bool, default=False,
        help="Whether to use sampling for initial generation."
    )
    parser.add_argument(
        "--temperature", type=float, default=0,
        help="If do_sample==True, Temperature for sampling."
    )
    parser.add_argument(
        "--top_p", type=float, default=0.9,
        help="Top-p for sampling."
    )
    parser.add_argument(
        "--use_vllm_for_initial_generation", action='store_true',
        help="Use vLLM for the initial solution generation as well."
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="GPU memory utilization factor for vLLM (0.0 to 1.0)."
    )
    parser.add_argument(
        "--non_reasoning_model", type=bool, default=False,
        help="non_reasoning_model")
    parser.add_argument(
        "--subset_index", type=int, default=0,
        help="Index of the subset to process (used for splitting the data across multiple runs).")
    parser.add_argument(
        "--num_subsets", type=int, default=1, 
        help="Total number of subsets to split the data into.")

    return parser.parse_args()


def plot_pruning_confidence(confidence_data, output_image_path="pruning_confidence_plot.png",window_size = 1):
    """
    生成一个散点图，可视化每个token的置信度，并在x轴上显示 "索引: 'token'"。
    当token数量过多时，会自动减少显示的标签数量以保持清晰。
    """
    if not confidence_data:
        print("No data available for plotting.")
        return

    # 1. 从 confidence_data 中提取绘图所需的数据
    token_indices = [p["token_index"] for p in confidence_data]
    confidences = [p["confidence"] for p in confidence_data]
    decoded_tokens = [p["decoded_token"] for p in confidence_data]

    if confidence_data and 'is_kept' in confidence_data[0]:
        # If the key exists, use the original blue/red logic based on its value.
        colors = ['blue' if p["is_kept"] else 'red' for p in confidence_data]
    else:
        # If the key is missing, create a list of 'black' for all data points.
        colors = ['black'] * len(confidence_data)

    # --- 计算滑动窗口平均值 ---
    smoothed_confidences = []
    if len(confidences) >= window_size:
        # 使用pandas高效计算滑动平均值，min_periods=1确保从序列开始就有平滑值
        series = pd.Series(confidences)
        smoothed_confidences = series.rolling(window=window_size, min_periods=1, center=True).mean().tolist()

    confidences = smoothed_confidences

    full_target_answer_str = ""
    if confidence_data:
        full_target_answer_str = confidence_data[0].get("confidence_target_token_string", "N/A")

    # 截断过长的答案字符串，以便在标题中显示
    display_target_str = full_target_answer_str
    if len(full_target_answer_str) > 50:
        display_target_str = full_target_answer_str[:47] + "..."

    # 2. 开始绘图
    plt.figure(figsize=(18, 10)) # 增加图表高度以容纳更长的x轴标签
    plt.scatter(token_indices, confidences, c=colors, alpha=0.7, s=50)
    
    # 设置图例
    handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Kept Token', markersize=10, markerfacecolor='blue', alpha=0.7),
        plt.Line2D([0], [0], marker='o', color='w', label='Pruned Token', markersize=10, markerfacecolor='red', alpha=0.7),
        plt.Line2D([0], [0], marker='o', color='w', label='Unkown Token', markersize=10, markerfacecolor='black', alpha=0.7),
    ]
    plt.legend(handles=handles, title="Token Status", loc="lower right")
    
    # 设置标题和y轴
    plt.xlabel("Token Index and Content") # 更新x轴标题
    plt.ylabel(f"Avg. Model Confidence for GT Answer Sequence ('{display_target_str}')")
    plt.title("Token Pruning Confidence Visualization", fontsize=16)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    plt.ylim(-0.05, 1.05)

    # 3. <--- 核心修改: 创建 "索引: 'token'" 格式的x轴标签 --->
    num_tokens = len(token_indices)
    
    if num_tokens > 40:  # 如果token太多，则抽样显示
        # 计算步长，目标是最多显示约10个标签，保持图表清爽
        step = max(1, num_tokens // 10)
        # 获取抽样后的索引和token
        tick_indices = token_indices[::step]
        # subsampled_tokens = decoded_tokens[::step]
        # 创建 "索引: 'token'" 格式的标签
        tick_labels = [f"{idx}\n{''.join(decoded_tokens[idx-3:idx+3])}" for idx in tick_indices]
        
        plt.xticks(ticks=tick_indices, labels=tick_labels, rotation=0, fontdict={'fontsize': 8})

    else: # 如果token数量不多，则全部显示
        # 创建 "索引: 'token'" 格式的标签
        tick_labels = [f"{idx}\n{decoded_tokens[idx]}" for idx in tick_indices]

        
        plt.xticks(ticks=token_indices, labels=tick_labels, rotation=0, fontdict={'fontsize': 9})
    
    # 确保x轴的范围能完整显示所有点
    if token_indices:
        plt.xlim(token_indices[0] - 1, token_indices[-1] + 1)
        
    plt.yticks(fontsize=10)
    plt.tight_layout() # 自动调整布局，防止标签被裁切
    
    # 4. 保存图像
    output_directory = os.path.dirname(output_image_path)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
        print(f"Created directory: {output_directory}")
    try:
        plt.savefig(output_image_path, dpi=300) # 使用高分辨率保存
        # print(f"Pruning confidence plot saved to {output_image_path}")
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()



def plot_pruning_confidence_12(confidence_data, unique_points=None, output_image_path="pruning_confidence_plot.png",window_size = 1):
    """
    生成一个散点图，可视化每个token的置信度，并在x轴上显示 "索引: 'token'"。
    当token数量过多时，会自动减少显示的标签数量以保持清晰。
    """
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.pyplot import MultipleLocator, FixedLocator
    from matplotlib.lines import Line2D
    from matplotlib import rcParams
    import matplotlib.ticker as mtick
    import matplotlib as mpl
    import seaborn as sns

    config = {
        "font.family": 'serif',
        "font.size": 11,  # Base font size, FONTSIZE variable will override for specific elements
        "mathtext.fontset": 'stix',
        "font.serif": ['Times New Roman'],
    }
    mpl.rc('pdf', fonttype=42) 
    rcParams.update(config)
    FONTSIZE = 16
    ALLWIDTH = 1.5 # Linewidth for bar edges
    BAR_EDGE_WIDTH = ALLWIDTH 
    HATCH = ['/', '\\', 'x', '+', '.', '*', 'o', 'O', '|', '-'] 
    COLOR = ["#264653","#299D92","#8AB17C","#E8C56B","#E66F51"]


    if not confidence_data:
        print("No data available for plotting.")
        return

    # 1. --- Extract and Prepare Data ---
    token_indices = np.array([p["token_index"] for p in confidence_data])
    confidences = np.array([p["confidence"] for p in confidence_data])
    colors = ['#299D92' if p.get("is_kept", False) else '#E66F51' for p in confidence_data]

    # Smooth the confidence data for plotting
    if len(confidences) >= window_size:
        series = pd.Series(confidences)
        smoothed_confidences = series.rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()
    else:
        smoothed_confidences = confidences

    # 2. --- Setup Figure ---
    plt.figure(figsize=(6.35 * 0.75, 3.65))

    # 3. --- Plotting Logic ---
    if not unique_points or len(unique_points) < 2:
        plt.scatter(token_indices, smoothed_confidences, color=colors, marker='o', s=15, alpha=1)
        handles = [
            Line2D([0], [0], marker='o', color='w', label='Retained', markerfacecolor='#299D92', markersize=14),
            Line2D([0], [0], marker='o', color='w', label='Pruned', markerfacecolor='#E66F51', markersize=14)
        ]
        plt.legend(handles=handles, loc="lower right")

    else:
        # --- Advanced Plotting with Rise/Fall/Early-Stop Markers ---
        max_analysis_idx = max(p['index'] for p in unique_points)
        # just For gsm8k qwen14B example 2
        analysis_len = min(max_analysis_idx + 7, len(smoothed_confidences))
        boundary_indices = sorted(list(set([0] + [p['index'] for p in unique_points if p['index'] < analysis_len] + [analysis_len - 1])))

        rise_data = {'indices': [], 'confidences': [], 'colors': []}
        fall_data = {'indices': [], 'confidences': [], 'colors': []}

        for i in range(len(boundary_indices) - 1):
            start_idx, end_idx = boundary_indices[i], boundary_indices[i+1]
            s_indices = token_indices[start_idx:end_idx + 1]
            s_confidences = smoothed_confidences[start_idx:end_idx + 1]
            s_colors = colors[start_idx:end_idx + 1]
            if len(s_confidences) < 2: continue
            
            target_data = rise_data if s_confidences[-1] >= s_confidences[0] else fall_data
            target_data['indices'].extend(s_indices)
            target_data['confidences'].extend(s_confidences)
            target_data['colors'].extend(s_colors)
        
        # Plot the classified segments
        # if rise_data['indices']:
        #     plt.scatter(rise_data['indices'], rise_data['confidences'], color=rise_data['colors'], marker='*', s=50, alpha=1)
        # if fall_data['indices']:
        #     plt.scatter(fall_data['indices'], fall_data['confidences'], color=fall_data['colors'], marker='v', s=20, alpha=1)


        if fall_data['indices']:
            plt.scatter(fall_data['indices'], fall_data['confidences'], color="#E66F51", marker='v', s=40, alpha=1)
        if rise_data['indices']:
            plt.scatter(rise_data['indices'], rise_data['confidences'], color="#299D92", marker='*', s=100, alpha=1)
        # Plot any data after the analysis with a star marker '*'
        if analysis_len < len(token_indices):
            plt.scatter(token_indices[analysis_len:], smoothed_confidences[analysis_len:], color=colors[analysis_len:], marker='o', s=20, alpha=1)

        # --- Setup a single, combined legend for all 5 states ---
        handles = [
            # Line2D([0], [0], marker='*', color='w', label='Retained Rise', markerfacecolor='#299D92', markersize=14, linestyle='None'),
            # Line2D([0], [0], marker='v', color='w', label='Retained Fall', markerfacecolor='#299D92', markersize=8, linestyle='None'),
            Line2D([0], [0], marker='*', color='w', label='Rise', markerfacecolor='#299D92', markersize=16, linestyle='None'),
            Line2D([0], [0], marker='v', color='w', label='Fall', markerfacecolor='#E66F51', markersize=10, linestyle='None'),
            Line2D([0], [0], marker='o', color='w', label='Early Stop', markerfacecolor='#E66F51', markersize=10, linestyle='None')
        ]
        plt.legend(handles=handles, loc="lower right", fontsize=FONTSIZE,
                ncol=1,
                handleheight=0.5,
                labelspacing=0.1,
                handlelength=0.8,
                handletextpad=0.1,
                columnspacing=1,
                borderpad=0.3,)
        # fig.legend(
        # handles, 
        # labels,
        # fontsize=FONTSIZE - 2,
        # loc='upper center', 
        # ncol=2,
        # handleheight=0.7,
        # labelspacing=0.2,
        # handlelength=1,
        # handletextpad=0.2,
        # columnspacing=1,
        # borderpad=0.3,
        # frameon=True,
        # bbox_to_anchor=(0.5, 0.87)
        # )

    # 4. --- Final Touches and Save ---
    plt.xlabel("Token Index",fontsize=FONTSIZE)
    plt.ylabel("Confidence",fontsize=FONTSIZE)
    plt.xticks(fontsize=FONTSIZE)
    plt.yticks(fontsize=FONTSIZE)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray', alpha=0.7)
    plt.ylim(-0.05, 1.05)
    if token_indices.any():
        plt.xlim(token_indices[0] - 1, token_indices[-1] + 1)
    plt.tight_layout()

    output_directory = os.path.dirname(output_image_path)
    if output_directory and not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
    try:
        plt.savefig(output_image_path, dpi=300)
    except Exception as e:
        print(f"Error saving plot: {e}")
    plt.close()


def collect_token_confidence_data(
    original_thinking: str,
    prompt_for_model: str,
    final_answer_string: str,
    vllm_engine: LLM,
    tokenizer: AutoTokenizer,
    top_k_factor: float = 1.1,
    max_model_len: int = 4096,
    logprobs_to_request: int = 20
) -> List[Dict[str, Any]]:
    """
    Generates confidence data for each token in the original thinking process.

    This function interacts with the vLLM engine to probe the model at each step
    of the thinking process and calculates the confidence that the model can predict
    the final answer. It also checks if the target answer tokens meet a top-k
    confidence criterion.

    Args:
        original_thinking: The original thought process string from the LLM.
        prompt_for_model: The initial prompt given to the LLM.
        final_answer_string: The ground truth final answer (e.g., "A", "12.5").
        vllm_engine: The initialized vLLM LLM engine.
        tokenizer: The tokenizer.
        top_k_factor: The factor to determine if a top-1 token is significantly
                      more probable than the top-2 token.
        max_model_len: The maximum model length from the tokenizer.
        logprobs_to_request: The number of top logprobs to request from vLLM.

    Returns:
        A list of dictionaries (confidence_data), where each dictionary contains
        statistics for a single token from the original thinking.
    """
    confidence_data = []
    
    original_thinking_token_ids = tokenizer.encode(original_thinking, add_special_tokens=False)


    # --- Edge Case Handling ---
    if not original_thinking_token_ids:
        print("Warning (Data Collection): Original thinking is empty.")
        return []

    if not final_answer_string.strip():
        confidence_data.append({
            "token_index": -1, "token_id": '', "decoded_token": '',
            "confidence": 0.0, "all_top_k_met": False,
            "confidence_target_token_string": final_answer_string
        })
        print("Warning (Data Collection): final_answer_string is empty.")
        for i, token_id in enumerate(original_thinking_token_ids):
            confidence_data.append({
                "token_index": i, "token_id": token_id,
                "decoded_token": tokenizer.decode([token_id]),
                "confidence": 0.0, "all_top_k_met": False
            })
        return confidence_data

    encoded_final_answer_gt = tokenizer.encode(final_answer_string, add_special_tokens=False)
    if not encoded_final_answer_gt:
        print("Warning (Data Collection): final_answer_string is empty after tokenization.")
        # Similar handling as empty final_answer_string
        confidence_data.append({
            "token_index": -1, "token_id": '', "decoded_token": '',
            "confidence": 0.0, "all_top_k_met": False,
            "confidence_target_token_string": final_answer_string
        })
        for i, token_id in enumerate(original_thinking_token_ids):
            confidence_data.append({
                "token_index": i, "token_id": token_id,
                "decoded_token": tokenizer.decode([token_id]),
                "confidence": 0.0, "all_top_k_met": False
            })
        return confidence_data


    
    
    # --- vLLM and Prompt Setup ---
    base_probe_stimulus_text = " The correct answer is "
    vllm_sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=logprobs_to_request
    )

    # --- Main Loop for Data Collection ---
    for i in range(-1, len(original_thinking_token_ids)):
        if i == -1:
            # This is the base case before any thinking tokens are added
            current_token_id = ''
            current_thinking_prefix_text = ''
            decoded_token = ''
        else:
            current_token_id = original_thinking_token_ids[i]
            current_thinking_prefix_text = tokenizer.decode(original_thinking_token_ids[:i+1])
            decoded_token = tokenizer.decode([current_token_id])

        # --- Batch Probe Generation ---
        batched_probe_prompts = []
        batched_target_gt_token_ids = []
        for j in range(len(encoded_final_answer_gt)):
            target_gt_token = encoded_final_answer_gt[j]
            # skip space
            if tokenizer.decode([target_gt_token]).isspace():
                continue
            if j == 0:
                probe_core_text = base_probe_stimulus_text
            else:
                decoded_gt_prefix = tokenizer.decode(encoded_final_answer_gt[:j])
                probe_core_text = base_probe_stimulus_text + decoded_gt_prefix
            
            prompt_text = f"{prompt_for_model}{current_thinking_prefix_text}{probe_core_text}"
            
            # Length check
            if len(tokenizer.encode(prompt_text, add_special_tokens=False)) >= max_model_len - 1:
                batched_probe_prompts.append(None) # Placeholder for skipped prompt
            else:
                batched_probe_prompts.append(prompt_text)
            batched_target_gt_token_ids.append(target_gt_token)

        # --- vLLM Inference ---
        valid_prompts = [p for p in batched_probe_prompts if p is not None]
        outputs_vllm = []
        if valid_prompts:
            try:
                outputs_vllm = vllm_engine.generate(valid_prompts, vllm_sampling_params, use_tqdm=False)
            except Exception as e:
                print(f"Error: vLLM generate failed during data collection: {e}")

        # --- Process Results and Calculate Metrics ---
        processed_confidences = []
        top_k_conditions_met = []
        output_idx = 0

        for k in range(len(batched_probe_prompts)):
            target_gt_token_id = batched_target_gt_token_ids[k]
            
            # Handle skipped or failed probes
            if batched_probe_prompts[k] is None or not outputs_vllm or output_idx >= len(outputs_vllm):
                processed_confidences.append(0.0)
                top_k_conditions_met.append(False)
                continue

            request_output = outputs_vllm[output_idx]
            output_idx += 1
            
            current_probe_prob = 0.0
            top_k_j_flag = False

            if request_output.outputs and request_output.outputs[0].logprobs:
                logprobs_dict = request_output.outputs[0].logprobs[0]
                
                if target_gt_token_id in logprobs_dict:
                    current_probe_prob = np.exp(logprobs_dict[target_gt_token_id].logprob)
                else:
                    # If target is not in top-k, its probability is effectively zero for our purpose
                    current_probe_prob = 0.0
                
                # Check top-k condition
                if current_probe_prob > 0:
                    sorted_logprobs = sorted([(np.exp(lp.logprob), t_id) for t_id, lp in logprobs_dict.items()], reverse=True)
                    if sorted_logprobs[0][1] == target_gt_token_id:
                        if len(sorted_logprobs) == 1 or sorted_logprobs[0][0] > top_k_factor * sorted_logprobs[1][0]:
                            top_k_j_flag = True
            
            processed_confidences.append(current_probe_prob)
            top_k_conditions_met.append(top_k_j_flag)

        # --- Store data for the current thinking token ---
        avg_confidence = np.mean(processed_confidences) if processed_confidences else 0.0
        confidence_data.append({
            "token_index": i,
            "token_id": current_token_id,
            "decoded_token": decoded_token,
            "confidence": avg_confidence,
            "all_top_k_met": all(top_k_conditions_met) if top_k_conditions_met else False
        })

    return confidence_data



def is_contextual_boundary(k: int, data: List[Dict[str, Any]], limit: int, end_tokens: List[str]) -> bool:
    """
    一个具有上下文感知能力的辅助函数，用于判断一个 token 是否是真正的边界。

    Args:
        k: 当前 token 在 data 列表中的索引。
        data: 包含所有 token 信息的列表。
        limit: 搜索的上限（通常是列表长度）。
        end_tokens: 句子结束标记的列表。
    
    Returns:
        如果 token 是一个真实的边界，则为 True，否则为 False。
    """
    token_text = data[k]['decoded_token']

    # e.g. 3.1
    if '.' in token_text:
        if token_text.strip() == '.' and k - 1 > 0 and k + 1 < limit:
            tmp_text = "".join([data[k-1]['decoded_token'],token_text,data[k+1]['decoded_token']])
            if re.search(r'\d\.\d', tmp_text):
                # 如果匹配成功 (e.g., 在 "3.14" 中找到了 "3.1")，说明是小数点
                return False  # 不是边界
            else:
                return True
        else:
            return True

    # --- 次要逻辑：处理其他无歧义的标点 ---
    # 如果代码执行到这里，说明 token 中不含 '.'，我们可以安全地进行简单检查。
    other_punctuations = [p for p in end_tokens if p != '.']
    if any(p in token_text for p in other_punctuations):
        return True

    return False

def find_sentence_boundary_idx(data: List[Dict[str, Any]], limit: int, end_tokens: List[str], current_idx: int, forward: bool):
    if forward:
        # Search forward for a sentence end
        for k in range(current_idx, limit):
            if is_contextual_boundary(k, data, limit, end_tokens):
                return k + 1  # Include the punctuation and the token after
        return limit # If no end found, go to the limit
    else:
        # Search backward for a sentence start
        for k in range(min(current_idx, limit - 1), -1, -1):
            if is_contextual_boundary(k, data, limit, end_tokens):
                return k + 1 # Start after the punctuation
        return 0 # If no start found, go to the beginning


def prune_by_rise_over_fall(
    confidence_data: List[Dict[str, Any]],
    tokenizer: "AutoTokenizer",
    smoothing_window: int = 10,
    keep_final_rise: bool = True,
    early_stop_threshold: float = 0.99,
    window_size: int = 10,
    peak_prominence: float = 0.085,
    rise_magnitude: float = 0.01,
    only_early_stop: int = 0,
    is_skeptical: int = 1
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    根据“上升收益 > 下降成本”的原则剪枝Token，并增加了高置信度平稳区的提前终止逻辑。

    Args:
        confidence_data: 包含置信度信息的字典列表。
        tokenizer: 分词器。
        smoothing_window: 用于平滑置信度曲线以消除噪声的滑动窗口大小。
        keep_final_rise: 是否无条件保留导向全局最高置信度的最后一个上升段（在未触发提前终止时生效）。
        early_stop_threshold: 触发提前终止检查的置信度阈值。
        window_size: 连续多少个token的置信度高于阈值时，触发提前终止。

    Returns:
        A tuple containing:
        - The pruned thinking process as a string.
        - The confidence_data list, now updated with the final "is_kept" decision.
    """
    sentence_end_tokens = ["。", "，", ".", ",", "?", "!", ";","；","\n","\t"]

    if not confidence_data:
        return "", [],[]

    if confidence_data[0]["token_index"] == -1:
        previous_avg_confidence = confidence_data[0]["confidence"]
        del confidence_data[0]
    else:
        previous_avg_confidence = 0 


    is_kept = [False] * len(confidence_data)
    confidences = [d['confidence'] for d in confidence_data]

    # --- 1. 寻找高置信度平稳区 (High-Confidence Plateau) ---
    plateau_start_index = -1
    if len(confidences) >= window_size:
        for i in range(len(confidences) - window_size + 1):
            window_slice = confidences[i : i + window_size]
            # 检查窗口内所有token的置信度是否都高于阈值
            if all(c >= early_stop_threshold for c in window_slice):
                plateau_start_index = i
                break  # 找到第一个满足条件的平稳区就停止搜索

    # 因为画图就是平滑了一下，上面i没有平滑，平滑后的idx对应plateau_start_index + 0.5window_size，
    truncation_index = len(confidences)
    if plateau_start_index != -1:
        # 1. 计算寻找句子结尾的起始点
        # 从平稳区开始后的半个窗口位置启动搜索
        search_start_point = min(len(confidences), plateau_start_index + int(0.5 * window_size))
        
        # 2. 从起始点开始，寻找第一个句子结束的标点
        # (需要能访问到生成的 token 列表, 这里假设它名为 generated_tokens)
        truncation_index = find_sentence_boundary_idx(
            confidence_data, len(confidences), sentence_end_tokens, search_start_point, forward=True
        )

        plateau_text_start_index = find_sentence_boundary_idx(
            confidence_data, len(confidences), sentence_end_tokens, plateau_start_index, forward=False
        )


        # truncation_index = -1 # 初始化截断点索引
        # found_sentence_end = False # Flag to indicate if a sentence end token is found
        # for i in range(search_start_point, len(confidences)):

        #     for j in confidence_data[i]['decoded_token']:
        #         if j in sentence_end_tokens:
        #             truncation_index = i + 1  # 加 1 以包含这个标点符号本身
        #             found_sentence_end = True # Set the flag
        #             break # Break inner loop (for j)
        #     if found_sentence_end: # Check the flag after inner loop
        #         break # Break outer loop (for i)
        # # 如果在 search_start_point 之后没有找到任何结束标点，则保留到最后
        # if truncation_index == -1:
        #     truncation_index = len(confidences)

        # 3. 根据找到的截断点更新 is_kept 列表
        for i in range(plateau_text_start_index, min(len(confidences), truncation_index)):
            is_kept[i] = True

        for i in range(min(len(confidences), truncation_index), len(confidences)):
            is_kept[i] = False

        # 我们只需要对平稳区之前的数据应用V形分析逻辑
        analysis_data_len = plateau_start_index
    else:
        # 如果没有找到平稳区，则对全部数据进行分析
        analysis_data_len = len(confidences)


    if only_early_stop:
        for j in range(0, min(len(confidences), truncation_index)):
            is_kept[j] = True



    # 如果分析区长度为0，则无需进行后续分析
    if analysis_data_len == 0:
        # 直接根据is_kept数组构建结果并返回
        kept_token_ids = [d["token_id"] for i, d in enumerate(confidence_data) if is_kept[i]]
        pruned_thinking = tokenizer.decode(kept_token_ids, skip_special_tokens=True)
        return pruned_thinking, confidence_data, [{'type': 'peak', 'index': 0}]

    # --- 2. 对“分析区”进行平滑和转折点分析 ---
    confidences_to_analyze = confidences[:analysis_data_len]
    
    # 数据平滑
    if len(confidences_to_analyze) >= smoothing_window:
        series = pd.Series(confidences_to_analyze)
        smoothed_confidences = series.rolling(window=window_size, min_periods=1, center=True).mean().to_numpy()
    else:
        smoothed_confidences = np.array(confidences_to_analyze)

    
    # 使用 prominence 参数寻找重要的波峰和波谷
    # 寻找波谷等同于在反转的序列中寻找波峰

    # if smoothed_confidences[0] < 0.5:
    #     find_peak_tmp_conf = [1] + list(smoothed_confidences)
    # else:
    #     find_peak_tmp_conf = [0] + list(smoothed_confidences)


    # if smoothed_confidences[-1] < 0.5:
    #     find_peak_tmp_conf = find_peak_tmp_conf + [1]
    # else:
    #     find_peak_tmp_conf = find_peak_tmp_conf + [0]
    # find_peak_tmp_conf = np.array(find_peak_tmp_conf)
    # smoothed_confidences = np.array([1,1,0.5,1,0,0,0,0,])


    peak_indices, peak_bases = find_peaks(smoothed_confidences, prominence=peak_prominence)
    valley_indices, valley_bases = find_peaks(-smoothed_confidences, prominence=peak_prominence)
    # 将找到的波峰和波谷合并，并添加首尾作为边界
    turning_points = []
    for idx in peak_indices:
        turning_points.append({'type': 'peak', 'index': idx})
    for idx in valley_indices:
        turning_points.append({'type': 'valley', 'index': idx})
    
    # turning_points = []
    # for idx in range(len(peak_bases["left_bases"])):
    #     turning_points.append({'type': 'valley', 'index': peak_bases["left_bases"][idx]})
    #     turning_points.append({'type': 'valley', 'index': peak_bases["right_bases"][idx]})
    # for idx in range(len(valley_bases["left_bases"])):
    #     turning_points.append({'type': 'peak', 'index': valley_bases["left_bases"][idx]})
    #     turning_points.append({'type': 'peak', 'index': valley_bases["right_bases"][idx]})

    # 按索引排序
    turning_points.sort(key=lambda p: p['index'])


    all_points = []
    for j in turning_points:
        if j['index'] >= 0 and j['index'] < len(smoothed_confidences):
            all_points.append(j)

 
    unique_points = []
    if all_points:
        unique_points.append(all_points[0])
        for i in range(1, len(all_points)):
            if all_points[i]['index'] > unique_points[-1]['index']:
                unique_points.append(all_points[i])


    # print(unique_points)
    # 716MODIFIED edge 
    if len(unique_points)==0:
        unique_points.append({'type': 'valley', 'index': 0})
        unique_points.append({'type': 'peak', 'index': len(smoothed_confidences)-1})

    else:
        if plateau_start_index!=-1 and unique_points[-1]['index']<len(smoothed_confidences)-window_size:
            unique_points.append({'type': 'peak', 'index': len(smoothed_confidences)-1})

        if unique_points[0]['index']>len(smoothed_confidences)+window_size:
            if unique_points[0]['type']=="valley":
                unique_points.append({'type': 'peak', 'index': 0})
            else:
                unique_points.append({'type': 'valley', 'index': 0})


    # print(unique_points)
    
    # exit()
    # --- 3. 在“分析区”应用“上升 > 下降”规则 ---
    # `segments_to_evaluate` will store pairs of (type, start_idx, end_idx, magnitude)
    # type: 'rise' or 'fall'
    segments_to_evaluate = []
    
    # If there are no unique points or only one, no meaningful segments can be formed.
    if len(unique_points) < 2:
        # If there's an initial rise from 0 to the single point (if it's a peak)
        if unique_points and unique_points[0]['type'] == 'peak' and unique_points[0]['index'] > 0:
            segments_to_evaluate.append({
                'type': 'rise',
                'start_idx': 0,
                'end_idx': unique_points[0]['index'],
                'magnitude': smoothed_confidences[unique_points[0]['index']] - smoothed_confidences[0]
            })
    else:
        # Initialize the start of the first segment
        current_segment_start_idx = unique_points[0]['index']
        
        # Add a virtual point at index 0 if the first unique point is not at 0
        if current_segment_start_idx > 0:
            # Determine if this virtual segment from 0 to first_point is a rise or fall
            if smoothed_confidences[0] <= smoothed_confidences[current_segment_start_idx]: # Assume rise or flat
                 segments_to_evaluate.append({
                    'type': 'rise',
                    'start_idx': 0,
                    'end_idx': current_segment_start_idx,
                    'magnitude': smoothed_confidences[current_segment_start_idx] - smoothed_confidences[0]
                })
            else: # Assume fall
                 segments_to_evaluate.append({
                    'type': 'fall',
                    'start_idx': 0,
                    'end_idx': current_segment_start_idx,
                    'magnitude': smoothed_confidences[0] - smoothed_confidences[current_segment_start_idx]
                })


        # print(unique_points)
        # exit()
        for i in range(len(unique_points) - 1):
            start_point = unique_points[i]
            end_point = unique_points[i+1]
            
            segment_type = None
            magnitude = 0

            if start_point['type'] == 'valley' and end_point['type'] == 'peak':
                segment_type = 'rise'
                magnitude = smoothed_confidences[end_point['index']] - smoothed_confidences[start_point['index']]
            elif start_point['type'] == 'peak' and end_point['type'] == 'valley':
                segment_type = 'fall'
                magnitude = smoothed_confidences[start_point['index']] - smoothed_confidences[end_point['index']]
            
            # Add segment only if it's a clear rise or fall
            if segment_type:
                segments_to_evaluate.append({
                    'type': segment_type,
                    'start_idx': start_point['index'],
                    'end_idx': end_point['index'],
                    'magnitude': magnitude
                })

    # ablation: kept all rise
    # if not is_skeptical:
    #     rise_magnitude = -1

    last_fall_magnitude = 0
    for i, segment in enumerate(segments_to_evaluate):
        segment_start_idx = segment['start_idx']
        segment_end_idx = segment['end_idx']

        if segment['type'] == 'rise':
            current_rise_magnitude = segment['magnitude']

            if current_rise_magnitude > last_fall_magnitude + rise_magnitude:
                is_kept_current_segment = True
            else:
                is_kept_current_segment = False
            
            if is_kept_current_segment:
                # Expand to sentence boundaries
                # If the previous segment was kept and adjacent, extend from its start.
                # Otherwise, find sentence start for current segment.
                token_swift = 0
                # if (segment_end_idx - segment_start_idx) > window_size:
                #     token_swift = int(0.5*window_size)

                sentence_start_idx = find_sentence_boundary_idx(
                    confidence_data, analysis_data_len, sentence_end_tokens, segment_start_idx + token_swift, forward=False
                )
                sentence_end_idx = find_sentence_boundary_idx(
                    confidence_data, analysis_data_len, sentence_end_tokens, segment_end_idx - token_swift, forward=True
                )


                # tmp = []
                for j in range(sentence_start_idx, sentence_end_idx):
                    is_kept[j] = True
                    # tmp.append(confidence_data[j]["decoded_token"])
                # print(f"Province:----{sentence_start_idx}----{sentence_end_idx}--------")
                # print("".join(tmp))
                # print(f"limit---{segment_start_idx}----{segment_end_idx}-")
                # tmp = []
                for j in range(segment_start_idx, segment_end_idx):
                    is_kept[j] = True
                #     tmp.append(confidence_data[j]["decoded_token"])
                # print("".join(tmp))
                
                # print("----------")
                
                if is_skeptical and i > 0 and segments_to_evaluate[i-1]['type'] == 'fall':
                    pre_fall_start_idx = segments_to_evaluate[i-1]['start_idx'] 
                    pre_fall_end_idx = segments_to_evaluate[i-1]['end_idx']

                    sentence_start_idx = find_sentence_boundary_idx(
                        confidence_data, analysis_data_len, sentence_end_tokens, pre_fall_start_idx + token_swift, forward=False
                    )
                    sentence_end_idx = find_sentence_boundary_idx(
                        confidence_data, analysis_data_len, sentence_end_tokens, pre_fall_end_idx - token_swift, forward=True
                    )
                    # tmp = []
                    for j in range(sentence_start_idx, sentence_end_idx):
                        is_kept[j] = True
                    #     tmp.append(confidence_data[j]["decoded_token"])
                    # print(f"Skeppr:----{sentence_start_idx}----{sentence_end_idx}----")
                    # print("".join(tmp))
                    # print(f"limit_skep---{pre_fall_start_idx}---{pre_fall_end_idx}--")
                    # tmp = []
                    for j in range(pre_fall_start_idx, pre_fall_end_idx):
                        is_kept[j] = True
                    #     tmp.append(confidence_data[j]["decoded_token"])
                    # print("".join(tmp))
                    
                    # print("----------")
                
                # Reset last_fall_magnitude after a kept rise
                last_fall_magnitude = 0 

        elif segment['type'] == 'fall':
            last_fall_magnitude = segment['magnitude']
            # Fall segments themselves are not marked as kept by this rule.
            # They only set the fall_magnitude for the next rise.





    # --- 4. (仅在未触发提前终止时) 保留最后的上升段 ---
    if keep_final_rise and plateau_start_index == -1 and confidences_to_analyze:
        max_conf_index_in_analysis = np.argmax(confidences_to_analyze)
        last_valley_idx = 0
        for point in reversed(unique_points):
            if point['type'] == 'valley' and point['index'] < max_conf_index_in_analysis:
                last_valley_idx = point['index']
                break
        for i in range(last_valley_idx, analysis_data_len):
            is_kept[i] = True


    # --- 5. 构建最终结果 ---
    # kept_token_ids = []
    # for i, kept in enumerate(is_kept):
    #     confidence_data[i]["is_kept"] = kept
    #     if kept:
    #         kept_token_ids.append(confidence_data[i]["token_id"])

    # pruned_thinking = tokenizer.decode(kept_token_ids, skip_special_tokens=True)
    # --- 5. 构建最终结果 (用 "..." 替换剪枝部分) 2025.12.1---
    result_parts = []
    current_segment_tokens = []
    current_segment_is_kept = None

    for i, kept in enumerate(is_kept):
        confidence_data[i]["is_kept"] = kept
        
        if current_segment_is_kept is None:
            current_segment_is_kept = kept
        
        if kept == current_segment_is_kept:
            current_segment_tokens.append(confidence_data[i]["token_id"])
        else:
            # 状态变化，处理之前的 segment
            if current_segment_is_kept:
                result_parts.append(tokenizer.decode(current_segment_tokens, skip_special_tokens=True))
            else:
                result_parts.append("...")
            
            # 开始新 segment
            current_segment_tokens = [confidence_data[i]["token_id"]]
            current_segment_is_kept = kept

    # 处理最后一个 segment
    if current_segment_tokens:
        if current_segment_is_kept:
            result_parts.append(tokenizer.decode(current_segment_tokens, skip_special_tokens=True))
        else:
            result_parts.append("...")

    pruned_thinking = "".join(result_parts)
    
    
    # maybe \n</think>
    if not pruned_thinking.endswith('\n'):
        pruned_thinking += '\n'

    if pruned_thinking.startswith(' '):
        pruned_thinking = pruned_thinking[1:]

    return pruned_thinking, confidence_data, unique_points

def main_example():
    args = parse_arguments_for_example()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
        
    tensor_parallel_size = len(args.cuda_visible_devices.split(','))
    print(f"Loading tokenizer from: {args.model_path}")
    try:
        # For Qwen models, trust_remote_code=True is often necessary for the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return


    dataset_handler = get_dataset(args.dataset_name)
    data, answer_type = dataset_handler.load_data()
    if "gsm8k" in args.dataset_name:
        data = data['train']
        if len(data) > 360:
            data = data.select(range(360))
    elif "math500" in args.dataset_name:
        data = data['test']
        if len(data) > 300:
            data = data.select(range(300))
    elif "aime24" in args.dataset_name:
        data = data['train']
        if len(data) > 100: data = data.select(range(100))
        args.tokenizer_default_max_length = args.tokenizer_default_max_length *2 # 8192*2
        args.max_new_tokens = args.max_new_tokens * 4 # 4096*4

    qa_data = dataset_handler.prepare_qa_data(data)
    # Split data into subsets
    total_questions = len(qa_data)
    subset_size = total_questions // args.num_subsets
    start_index = args.subset_index * subset_size
    end_index = total_questions if args.subset_index == args.num_subsets - 1 else (args.subset_index + 1) * subset_size
    qa_data_subset = {k: qa_data[k] for k in list(qa_data.keys())[start_index:end_index]}
    
    # Generate prompts with tokenizer
    prompts_for_model, qa_datas_subset = generate_prompt(args, logger, qa_data_subset, answer_type=answer_type, tokenizer=tokenizer)
    # print("-----------------")
    # print(prompts_for_model[0])
    # print("-----------------")
    # exit()
    # 存在think token: QwQ32 自动prompt<think>, Deepseek 自动prompt<think>, Qwen3 
    # 不存在: Qwen2.5
    THINK_TOKEN_STR = "<think>"
    THINK_END_TOKEN_STR = "</think>"

    try:
        # Check if these are actual tokens in the vocabulary
        if THINK_TOKEN_STR not in tokenizer.vocab or THINK_END_TOKEN_STR not in tokenizer.vocab:
            print(f"Warning: Standard Qwen think tokens ('{THINK_TOKEN_STR}', '{THINK_END_TOKEN_STR}') not in vocab.")
            # Fallback or error if essential special tokens are missing for the chosen model.
            # For demonstration, we'll try to encode, but this might lead to issues if they are not proper tokens.
            # If your model uses a different ID like 151668, you MUST define it here and its start pair.
            # e.g. THINK_END_ID = 151668; THINK_START_ID = <ID for your model's start think token>

        # We get IDs by encoding. add_special_tokens=False is important here.
        THINK_START_ID = tokenizer.encode(THINK_TOKEN_STR, add_special_tokens=False)
        THINK_END_ID = tokenizer.encode(THINK_END_TOKEN_STR, add_special_tokens=False)

        if not THINK_START_ID or not THINK_END_ID: # Should not happen if they are in vocab
            raise ValueError("Think start/end token strings did not encode to valid IDs.")
        
        print(f"THINK_START_ID(the first): {THINK_START_ID} ('{THINK_TOKEN_STR}'), THINK_END_ID: {THINK_END_ID} ('{THINK_END_TOKEN_STR}')")
        THINK_START_ID = THINK_START_ID[0] # Taking the first ID, assuming they are single tokens
        THINK_END_ID = THINK_END_ID[0]

    except Exception as e:
        print(f"Critical Error: Could not determine think token IDs for Qwen model: {e}")
        print("Please ensure your model supports Qwen-style thinking or adjust THINK_TOKEN_STR/THINK_END_TOKEN_STR.")


    current_max_len = getattr(tokenizer, 'model_max_length', None)
    condition_to_override = False
    reason_for_override = ""
    if current_max_len is None:
        condition_to_override = True; reason_for_override = "is not set"
    elif current_max_len > args.tokenizer_max_length_override_threshold:
        condition_to_override = True; reason_for_override = f"({current_max_len}) exceeds threshold ({args.tokenizer_max_length_override_threshold})"
    
    if condition_to_override:
        print(f"Warning: tokenizer.model_max_length {reason_for_override}. Setting to {args.tokenizer_default_max_length}.")
        tokenizer.model_max_length = args.tokenizer_default_max_length
    elif current_max_len:
        print(f"Using tokenizer.model_max_length: {tokenizer.model_max_length}")
    else: 
        print(f"Warning: tokenizer.model_max_length was None or in an unexpected state. Setting to {args.tokenizer_default_max_length}.")
        tokenizer.model_max_length = args.tokenizer_default_max_length

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print(f"Tokenizer pad_token_id was None, set to eos_token_id: {tokenizer.pad_token_id}")
        else: # Fallback if eos is also None, very unlikely for proper models
            tokenizer.pad_token_id = 0 
            print(f"Tokenizer pad_token_id and eos_token_id were None. Set pad_token_id to 0. This may cause issues.")


    print(f"\n--- Initializing vLLM Engine from: {args.model_path} ---")
    vllm_max_model_len = int(tokenizer.model_max_length) if tokenizer.model_max_length is not None else int(args.tokenizer_default_max_length)
    print(f"Attempting to use tensor_parallel_size: {tensor_parallel_size}, vLLM max_model_len: {vllm_max_model_len}")
    

    vllm_engine = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=vllm_max_model_len, 
        trust_remote_code=True, # Often needed for Qwen
        gpu_memory_utilization=args.gpu_memory_utilization,
        # enable_chunked_prefill=True,
        dtype="bfloat16",
        seed=1,
    )
    # Actual max model length used by vLLM engine
    effective_max_model_len = vllm_engine.llm_engine.model_config.max_model_len
    # print(f"vLLM Engine initialized. Effective Max model length: {effective_max_model_len}. Using {tensor_parallel_size} GPU(s).")

    gen_sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.do_sample else 0.0,
        top_p=args.top_p if args.do_sample else 0.1,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [],
        # We want the model to generate its full thought and answer.
    )
    


    for i, (prompt_for_model, question, answer_text) in enumerate(tqdm(zip(prompts_for_model, list(qa_datas_subset.keys()), list(qa_datas_subset.values())), total=len(prompts_for_model), desc="Processing dataset")):

        correct_answer = dataset_handler.extract_answer(answer_text)

        vllm_outputs = vllm_engine.generate([prompt_for_model], gen_sampling_params, use_tqdm=False) 
        
        generated_full_output_text = ""
        generated_sequence_ids = []

        if vllm_outputs and vllm_outputs[0].outputs:
            generated_full_output_text = vllm_outputs[0].outputs[0].text
            generated_sequence_ids = vllm_outputs[0].outputs[0].token_ids
        else:
            print("Error: vLLM did not return expected output.")
            return
            
        # print(f"\nGenerated Full Output (vLLM) (length {len(generated_sequence_ids)} tokens):\n{generated_full_output_text}")

        org_is_correct = dataset_handler.check(correct_answer, dataset_handler.extract_answer(generated_full_output_text))

        
        idx_of_qwen_think_end_id_in_generated = -1
        try:
            # Search for QWEN_THINK_END_ID in the newly generated tokens
            idx_of_qwen_think_end_id_in_generated = generated_sequence_ids.index(THINK_END_ID)
        except ValueError:
            print(f"\nERROR: Qwen's end think token ID ({THINK_END_ID} for '{THINK_END_TOKEN_STR}') not found in the generated sequence.")
            # The 'pruned_response_str' will remain as the original prompt + raw vLLM output.
            # No specific pruning will occur.
            
        if idx_of_qwen_think_end_id_in_generated != -1:
            # print(f"\nFound Qwen end think token ID at index {idx_of_qwen_think_end_id_in_generated} in the *generated* sequence.")
            
            # Extract token IDs for the thinking content (these are from the START of generated_sequence_ids)
            ids_thinking_content_generated = generated_sequence_ids[:idx_of_qwen_think_end_id_in_generated]
            text_thinking_content_to_prune = tokenizer.decode(ids_thinking_content_generated, skip_special_tokens=True)
            
            # Extract token IDs for content after the thinking block
            # ids_after_think_block_generated = generated_sequence_ids[idx_of_qwen_think_end_id_in_generated + 1:]
            
            # print(f"\nExtracted Thinking Content for Pruning (decoded, {len(ids_thinking_content_generated)} tokens):\n{text_thinking_content_to_prune[:500]}...")

            if not text_thinking_content_to_prune.strip():
                print("Warning: Content within Qwen think block (before end token) is empty or whitespace. Skipping pruning of this block.")
                # pruned_response_str remains as original prompt + raw vLLM output
            else:

                # print("\n--- Pruning Content Within Qwen Think Block (using vLLM) ---")

                # 步骤 1: 调用第一个函数，仅收集每个 token 的置信度数据
                # 这是计算密集型步骤，负责与 vLLM 交互
                collected_token_data = collect_token_confidence_data(
                    original_thinking=text_thinking_content_to_prune,
                    prompt_for_model=prompt_for_model,
                    final_answer_string=correct_answer,
                    vllm_engine=vllm_engine,
                    tokenizer=tokenizer,
                    top_k_factor=args.top_k_factor,
                    max_model_len=effective_max_model_len,
                )
                

                
                output_confidence_file = f"./step1_confidence_data/{args.dataset_name}/{args.model_path.split('/')[-1]}/{i}_{args.dataset_name}_confidence_data.jsonl"
                output_directory = os.path.dirname(output_confidence_file)
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory, exist_ok=True)
                    print(f"Created directory: {output_directory}")
                with open(output_confidence_file,"w") as f:
                    json.dump(collected_token_data, f, ensure_ascii=False, indent=4)


                # plot_pruning_confidence(collected_token_data,output_image_path="./plot_fig/"+str(i)+".png", window_size = args.window_size,)


                # 步骤 2: 调用第二个函数，根据收集到的数据进行剪枝
                # 这一步很快，不与 vLLM 交互，只应用剪枝逻辑
                # pruned_think_text, confidence_data_for_graph = prune_thinking_from_data(
                #     confidence_data=collected_token_data,
                #     tokenizer=tokenizer,
                #     growth_factor=args.growth_factor,
                #     top_k_growth_factor=args.top_k_growth_factor,
                #     window_size = args.window_size,
                # )
                
                pruned_think_text, confidence_data_for_graph,_ = prune_by_rise_over_fall(
                        confidence_data=collected_token_data,
                        tokenizer=tokenizer,
                        window_size = args.window_size,
                        peak_prominence=args.peak_prominence,
                        rise_magnitude = args.rise_magnitude,
                        only_early_stop = args.only_early_stop,
                        is_skeptical = args.is_skeptical
                )
                if i<10:
                    plot_pruning_confidence(confidence_data_for_graph, output_image_path="./plot_fig/"+str(i)+".png", window_size = args.window_size,)
                
                text_qwen_think_end_token = tokenizer.decode([THINK_END_ID], skip_special_tokens=False) # e.g., "</think>"
                # text_after_think_segment_generated = tokenizer.decode(ids_after_think_block_generated, skip_special_tokens=False)
                
                pruned_response_str = (
                    prompt_for_model +  # This string already contains the Qwen start-of-think commands
                    pruned_think_text +
                    text_qwen_think_end_token
                    # text_after_think_segment_generated
                )

                pruned_outputs = vllm_engine.generate([pruned_response_str], gen_sampling_params, use_tqdm=False) 
                pruned_outputs_str = pruned_outputs[0].outputs[0].text

                pruned_res_answer = dataset_handler.extract_answer(pruned_outputs_str)

                pruned_is_correct = dataset_handler.check(pruned_res_answer, correct_answer)
            
                data_to_save = {
                    "question": question,
                    "correct_answer_text": answer_text,
                    "correct_answer": correct_answer,
                    "org_think_text": generated_full_output_text,
                    "org_is_correct": org_is_correct,
                    'pruned_think_text': pruned_think_text,
                    "pruned_is_correct": pruned_is_correct,
                }

    
                # 追加写入 JSONL 文件
                # output_jsonl_file = f"./step2_compressed_data/{args.dataset_name}/{args.model_path.split('/')[-1]}/{args.dataset_name}_pruned_data.jsonl"
                output_directory = os.path.dirname(args.output_jsonl_file)
                # 创建目录（如果它还不存在的话）
                # exist_ok=True 表示如果目录已存在，则不会引发错误
                if not os.path.exists(output_directory):
                    os.makedirs(output_directory, exist_ok=True)
                    print(f"Created directory: {output_directory}")
                with open(args.output_jsonl_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(data_to_save, ensure_ascii=False) + '\n')

                # if i > 3:
                #     break   


        # 原始输出，剪枝前：generated_full_output_text
        # prompt输入：prompt_for_model
        # pruned_response_str
        # 剪枝后think：pruned_think_text
                

        # print(f"\nFinal Processed Response String (length {len(tokenizer.encode(pruned_response_str,add_special_tokens=False))} tokens):\n{pruned_response_str}")




if __name__ == "__main__":

    main_example()