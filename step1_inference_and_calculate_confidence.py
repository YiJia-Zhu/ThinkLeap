import argparse
import json
import os
import sys
from tqdm import tqdm
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.dataset_loader import *
from typing import List, Dict, Any, Tuple, Set
# export CUDA_VISIBLE_DEVICES=0,2
from vllm import LLM, SamplingParams
import random
import re
import pickle


# Initialize logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
seed = 1
torch.manual_seed(seed)
np.random.seed(seed)
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
        default="./merge_models/Qwen3-14B-gsm8k-25",
        help="Path to the Hugging Face model for generation and pruning."
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="2,3",
        help="Comma-separated list of GPU IDs to use (e.g., '0,1'). This determines the tensor_parallel_size."
    )

    

    ####### param
    parser.add_argument(
        "--window_size", type=int, default=10,
        help="for prune"
    )
    parser.add_argument(
        "--peak_prominence", type=float, default=0.03,
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
        "--non_answer_confidence", type=int, default=0,
        help="non_answer_confidence"
    )

    parser.add_argument(
        "--top_k_factor", type=float, default=2,
        help="Factor for top_k probability check in pruning strategy."
    )


    #########
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

def parse_generated_distractors(generated_text, dataset_handler, idx, correct_answer_text_list, num_expected = 3):
    """
    从LLM生成的文本中解析出干扰项列表。
    设计上尽可能稳健，以应对模型不完全按指令输出的情况。
    """

    if '</think>' in generated_text:
        generated_text = generated_text.split('</think>', 1)[1]

    lines = re.findall(r'[A-Da-d][\.\)]\s*(.*?)(?=\s*[A-Da-d][\.\)]|$)', generated_text, re.DOTALL)
    distractors_org = [
        line for line in lines 
        if line and "sure" not in line.lower() and "here are" not in line.lower() and "plausible" not in line.lower()
    ]
    distractors = []
    for d in distractors_org:
        distractor = re.sub(r'^[0-9A-Da-d][\.\)]\s*', '', d).strip()
        distractor = dataset_handler.extract_answer("\\boxed{" + distractor+ "}")
        if distractor is None:
            continue
        if len(distractor)>100:
            continue
        if "=" in distractor:
            distractor = distractor.split('=')[-1]

        if not dataset_handler.check(distractor, dataset_handler.extract_answer(correct_answer_text_list[idx])):
            distractors.append(distractor)

    if len(distractors) < num_expected:
        print(f"Warning: Parsed only {len(distractors)} distractors. Padding with defaults.")
        defaults = [dataset_handler.extract_answer(correct_answer_text_list[idx-i]) for i in range(num_expected - len(distractors))]
        distractors.extend(defaults)
    return distractors[:num_expected]

def create_final_mc_probe(
    correct_answer: str,
    generated_distractors: list[str],
    tokenizer: any
) -> (str, int):
    """使用预生成的干扰项创建最终的多选题探针文本和正确答案ID。"""
    choices = generated_distractors + [correct_answer]
    random.shuffle(choices)
    correct_choice_index = choices.index(correct_answer)
    choice_labels = [chr(ord('A') + i) for i in range(len(choices))]
    correct_choice_label = choice_labels[correct_choice_index]

    # mc_question_prompt_text = "\n\nBased on the reasoning, which of the following is the correct final answer?\n"
    mc_question_prompt_text = ""
    mc_prompt_text = "\n\nBased on the reasoning, I will select an option below:\n"


    for i, choice in enumerate(choices):
        mc_prompt_text += f"{choice_labels[i]}) {choice}\n"
    # mc_prompt_text = "\nNow I can directly output the single letter of the correct option. \nCorrect option (A, B, C, or D):"
    mc_prompt_text += "\nCorrect option (A, B, C, or D):"

    try:
        correct_choice_token_id = tokenizer.encode(f" {correct_choice_label}", add_special_tokens=False)[0]
    except (IndexError, ValueError):
        correct_choice_token_id = tokenizer.encode(correct_choice_label, add_special_tokens=False)[0]
    return mc_question_prompt_text, mc_prompt_text, correct_choice_token_id

def calculate_metrics_from_outputs_multi(
    probe_outputs: List[Any],
    target_gt_token_ids: List[int],
    top_k_factor: float
) -> (float, bool):
    """
    使用链式概率处理单个思考步骤的一批vLLM探针输出。
    
    Args:
        probe_outputs: 一个思考步骤对应的vLLM RequestOutput对象列表。
        target_gt_token_ids: 最终答案的基准真相（ground truth）词元ID列表。
        top_k_factor: 用于top-k优势检查的因子。

    Returns:
        一个元组，包含 (chained_probability, all_top_k_conditions_met)。
    """
    if not probe_outputs:
        return 0.0, False

    # 初始化总对数概率为0.0，这在对数空间中相当于普通空间中的乘法单位“1”。
    total_log_prob = 0.0
    top_k_conditions_met = []
    
    # --- 修改后的逻辑：使用对数概率实现链式相乘 ---
    for k, request_output in enumerate(probe_outputs):
        target_gt_token_id = target_gt_token_ids[k]
        top_k_j_flag = False
        
        if request_output.outputs and request_output.outputs[0].logprobs:
            logprobs_dict = request_output.outputs[0].logprobs[0]
            
            if target_gt_token_id in logprobs_dict:
                current_log_prob = logprobs_dict[target_gt_token_id].logprob
                # 将对数概率相加，而不是将普通概率相乘
                total_log_prob += current_log_prob

                # top-k条件的检查逻辑可以保留，因为它针对每个单独的词元
                # 为保持一致性和数值稳定性，这里的比较也在对数空间进行
                sorted_logprobs = sorted([(lp.logprob, t_id) for t_id, lp in logprobs_dict.items()], reverse=True)
                if sorted_logprobs[0][1] == target_gt_token_id:
                    if len(sorted_logprobs) == 1 or sorted_logprobs[0][0] > np.log(top_k_factor) + sorted_logprobs[1][0]:
                        top_k_j_flag = True
            else:
                # 如果序列中任何一个词元的概率为0（即不在logprobs字典中），
                # 那么整个序列的概率就为0。
                # 在对数空间中，这对应于负无穷大。
                total_log_prob = -np.inf
                break 

        top_k_conditions_met.append(top_k_j_flag)

    # 将最终的对数概率总和转换回正常的概率值
    chained_probability = np.exp(total_log_prob)
    
    all_top_k_met = all(top_k_conditions_met) if top_k_conditions_met else False
    
    return chained_probability, all_top_k_met

# The original `collect_token_confidence_data` function is now removed.
# Its logic is integrated into main_example and the helper function above.


def main_example():

    args = parse_arguments_for_example()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    tensor_parallel_size = len(args.cuda_visible_devices.split(','))

    # --- Tokenizer and Dataset Loading (unchanged) ---
    print(f"Loading tokenizer from: {args.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}"); return
    
    dataset_handler = get_dataset(args.dataset_name)
    data, answer_type = dataset_handler.load_data()
    # Data Subsetting...
    if "gsm8k" in args.dataset_name:
        data = data['train']
        if len(data) > 360: data = data.select(range(360))
    elif "math500" in args.dataset_name:
        data = data['test']
        if len(data) > 300: data = data.select(range(300))
        args.tokenizer_default_max_length = 10240 + 4096
        args.max_new_tokens = 10240 +2048
    elif "aime24" in args.dataset_name:
        data = data['train']
        if len(data) > 100: data = data.select(range(100))
        args.tokenizer_default_max_length = 10240 + 4096
        args.max_new_tokens = 10240 +2048
    elif "arc_challenge" in args.dataset_name:
        data = data['train']
        if len(data) > 300: data = data.select(range(300))
        args.tokenizer_default_max_length = 10240 + 4096
        args.max_new_tokens = 10240 +2048
    elif "amc23" in args.dataset_name:
        data = data['test']
        if len(data) > 100: data = data.select(range(100))
        args.tokenizer_default_max_length = 10240 + 4096
        args.max_new_tokens = 10240 +2048

    qa_data = dataset_handler.prepare_qa_data(data)
    # Subsetting logic for parallel processing...
    total_questions = len(qa_data)
    subset_size = total_questions // args.num_subsets
    start_index = args.subset_index * subset_size
    end_index = total_questions if args.subset_index == args.num_subsets - 1 else (args.subset_index + 1) * subset_size
    qa_data_subset = {k: qa_data[k] for k in list(qa_data.keys())[start_index:end_index]}
    prompts_for_model, qa_datas_subset = generate_prompt(args, logger, qa_data_subset, answer_type=answer_type, tokenizer=tokenizer)
    
    # a = dataset_handler.extract_answer("Answer: \\boxed{0.11}")
    # b = dataset_handler.extract_answer("Answer: \\boxed{11/100}")

    # print(dataset_handler.check(a,b))
    # print(a,b)
    # exit()

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
    effective_max_model_len = vllm_engine.llm_engine.model_config.max_model_len
    gen_sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens, temperature=args.temperature if args.do_sample else 0.0,
        top_p=args.top_p if args.do_sample else 0.1,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    )
    # --- ### PHASE 1: BATCHED INITIAL GENERATION ### ---
    print("\n--- Phase 1: Generating initial 'thinking' responses for all samples ---")
    model_name = args.model_path.split('/')[-1]
    cache_dir = f"./cache1_thinking_cache/{args.dataset_name}"
    cache_file = os.path.join(cache_dir, f"{model_name}_initial_outputs_subset_{args.subset_index}.pkl")

    # 检查缓存文件是否存在
    if os.path.exists(cache_file):
        print(f"Found cache. Loading initial_outputs from: {cache_file}")
        # 使用 pickle 从二进制文件加载复杂对象
        with open(cache_file, 'rb') as f:
            initial_outputs = pickle.load(f)
    else:
        initial_outputs = vllm_engine.generate(prompts_for_model, gen_sampling_params, use_tqdm=True)
        os.makedirs(cache_dir, exist_ok=True)
        # 使用 pickle 将对象以二进制格式写入文件
        with open(cache_file, 'wb') as f:
            pickle.dump(initial_outputs, f)

    # --- ### PHASE 1.5: PREPARE AND GENERATE DISTRACTORS (FOR LATEX SAMPLES ONLY) ### ---
    print("\n--- Phase 1.5: Generating distractors for LaTeX samples  ---")
    # TODO MATH500 train 273
    distractors_by_sample_index = {}
    if answer_type == "latex_compression":
        model_name = args.model_path.split('/')[-1]
        cache_dir = f"./cache2_distractor_cache/{args.dataset_name}"
        # Use the same one 
        cache_file = os.path.join(cache_dir, f"distractors_subset_{args.subset_index}.json")

        # 检查缓存文件是否存在
        if os.path.exists(cache_file):
            print(f"Found cache. Loading distractors from: {cache_file}")
            with open(cache_file, 'r', encoding='utf-8') as f:
                # JSON 的键是字符串，加载后需要转换回整数
                distractors_by_sample_index = {int(k): v for k, v in json.load(f).items()}
        else:

            print(f"No cache found at {cache_file}. Generating distractors...")

            distractor_gen_prompts = []
            latex_sample_indices = []
            DISTRACTOR_GEN_INSTRUCTION = (
                "\n\nPlease act as a math competition problem designer. Your task is to create 3 highly convincing but incorrect 'distractor' answers in LaTeX format for the problem above. A perfect distractor results from a common student error."
                "\n\nIMPORTANT: Your output must strictly follow the example format below, including the lettered prefixes (A, B, C) and newlines. Do not add any other text, thoughts, or explanations."
                "\n\n--- EXAMPLE START ---"
                "\nA) \\frac{3}{2}, x\\in[-2,7]"
                "\nB) \\text{ellipse}"
                "\nC) (0, 3)"
                "\n--- EXAMPLE END ---"
            ).strip()
            
            for i, prompt in enumerate(prompts_for_model):
                text_to_remove = ", and put the final answer in a single \\boxed{}, which must use latex_compression (e.g., \\frac{3}{2}, x\\in[-2,7], \\frac{20000}{\\pi},'ellipse'...)"
                prompt_no_box = prompt.replace(text_to_remove, DISTRACTOR_GEN_INSTRUCTION)
                prompt_no_box += "Now I konw the correct answer is" + list(qa_datas_subset.values())[i] + "\n And Then I need generate incorrect answers."
                distractor_gen_prompts.append(prompt_no_box)
                latex_sample_indices.append(i)

            if distractor_gen_prompts:
                print(f"Generating distractors for {len(distractor_gen_prompts)} LaTeX samples in one batch...")
                distractor_outputs = vllm_engine.generate(distractor_gen_prompts, gen_sampling_params, use_tqdm=True)
                
                for i, output in enumerate(distractor_outputs):
                    original_sample_idx = latex_sample_indices[i]
                    distractors_by_sample_index[original_sample_idx] = parse_generated_distractors(output.outputs[0].text, dataset_handler, i,list(qa_datas_subset.values()))
            
            # 将新生成的干扰项保存到缓存文件
            print(f"Saving generated distractors to cache: {cache_file}")
            os.makedirs(cache_dir, exist_ok=True)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(distractors_by_sample_index, f, ensure_ascii=False, indent=4)

    # --- ### PHASE 2: PREPARE ALL PROBE PROMPTS ### ---
    print("\n--- Phase 2: Preparing all confidence probe prompts ---")
    all_probe_prompts = []
    # This map is crucial for re-assembling the results later
    results_map = [] 


    if args.non_answer_confidence:
        tokens_of_interest = {
            "confident": tokenizer.encode("confident", add_special_tokens=False)[0],
            "sure": tokenizer.encode("sure", add_special_tokens=False)[0],
            "pretty": tokenizer.encode("pretty", add_special_tokens=False)[0]
        }

    if answer_type == "number":
        base_probe_stimulus_text = " The correct answer is "
    elif answer_type == "option letter":
        base_probe_stimulus_text = " Now I konw the correct option (A, B, C, or D). Answer:"

    for i, (prompt_for_model, question, answer_text) in enumerate(tqdm(zip(prompts_for_model, list(qa_datas_subset.keys()), list(qa_datas_subset.values())), total=len(prompts_for_model), desc="Preparing Probes")):

        correct_answer = dataset_handler.extract_answer(answer_text)
        initial_output = initial_outputs[i]
        
        sample_map_entry = {
            "sample_index": i, "prompt_for_model": prompt_for_model,
            "correct_answer": correct_answer, "thinking_steps": []
        }
        if not correct_answer: continue
        if not (initial_output.outputs and correct_answer.strip()): continue
        
        generated_sequence_ids = initial_output.outputs[0].token_ids
        try:
            end_think_idx = generated_sequence_ids.index(THINK_END_ID)
            thinking_token_ids = generated_sequence_ids[:end_think_idx]
        except ValueError:
            print(f"Warning: End think token not found for sample {i}. Skipping.")
            continue

        if answer_type == "latex_compression":
            # 从Phase 2准备好的字典中获取该样本的干扰项
            distractors = distractors_by_sample_index.get(i, ["\\text{Default A}", "\\text{Default B}", "\\text{Default C}"])
            mc_question_prompt_text , probe_core_text, target_id = create_final_mc_probe(correct_answer, distractors, tokenizer)
            # 构建完整的探针prompt，复用现有变量
            text_to_remove = ", and put the final answer in a single \\boxed{}, which must use latex_compression (e.g., \\frac{3}{2}, x\\in[-2,7], \\frac{20000}{\\pi},'ellipse'...)"
            prompt_for_model_no_box = prompt_for_model.replace(text_to_remove, mc_question_prompt_text)
        elif answer_type == "option letter":
            if correct_answer[0] == " ":
                tmp = correct_answer
            else:
                tmp = " "+correct_answer
            encoded_final_answer_gt = tokenizer.encode(tmp, add_special_tokens=False)
            if not encoded_final_answer_gt: continue
        else:
            encoded_final_answer_gt = tokenizer.encode(correct_answer, add_special_tokens=False)
            if not encoded_final_answer_gt: continue
           

        # Iterate through each step of the thinking process
        for t in range(-1, len(thinking_token_ids)):
            if t == -1:
                current_thinking_prefix_text = ''
                current_token_id = ''
                decoded_token = '' # Special marker for the pre-thinking state
            else:
                current_token_id = thinking_token_ids[t]
                current_thinking_prefix_text = tokenizer.decode(thinking_token_ids[:t+1])
                decoded_token = tokenizer.decode([current_token_id])

            prompts_for_this_step = []
            target_ids_for_this_step = []
            # For each thinking step, create a probe for each token of the final answer
            if answer_type == "latex_compression":            
                prompt_text = f"{prompt_for_model_no_box}{current_thinking_prefix_text}{probe_core_text}"

                prompts_for_this_step.append(prompt_text)
                target_ids_for_this_step.append(target_id)
            
            # 您原来的逻辑被完整保留在 ELSE 分支中，完全不受影响
            elif args.non_answer_confidence:
                prompt_text = f"{prompt_for_model}{current_thinking_prefix_text} So, I'm "
                prompts_for_this_step.append(prompt_text)
                prompts_for_this_step.append(prompt_text)
                prompt_text = f"{prompt_for_model}{current_thinking_prefix_text} So, I'm pretty "
                prompts_for_this_step.append(prompt_text)
                prompts_for_this_step.append(prompt_text)

                target_ids_for_this_step.append(tokens_of_interest["sure"])
                target_ids_for_this_step.append(tokens_of_interest["confidence"])
                target_ids_for_this_step.append(tokens_of_interest["sure"])
                target_ids_for_this_step.append(tokens_of_interest["confidence"])

            else:
                for j in range(len(encoded_final_answer_gt)):
                    target_gt_token = encoded_final_answer_gt[j]
                    # skip space
                    if tokenizer.decode([target_gt_token]).isspace():
                        continue
                    target_ids_for_this_step.append(target_gt_token)

                    if j == 0:
                        probe_core_text = base_probe_stimulus_text
                    else:
                        decoded_gt_prefix = tokenizer.decode(encoded_final_answer_gt[:j])
                        probe_core_text = base_probe_stimulus_text + decoded_gt_prefix
                    
                    prompt_text = f"{prompt_for_model}{current_thinking_prefix_text}{probe_core_text}"
                    prompts_for_this_step.append(prompt_text)

            all_probe_prompts.extend(prompts_for_this_step)
            sample_map_entry["thinking_steps"].append({
                "token_index": t, "token_id": current_token_id, "decoded_token": decoded_token,
                "num_probes": len(prompts_for_this_step),
                "target_gt_token_ids": target_ids_for_this_step
            })

        results_map.append(sample_map_entry)

    # --- ### PHASE 3: BATCHED PROBE INFERENCE ### ---
    print(f"\n--- Phase 3: Running inference on {len(all_probe_prompts)} probe prompts ---")
    if all_probe_prompts:
        probe_sampling_params = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
        all_probe_outputs = vllm_engine.generate(all_probe_prompts, probe_sampling_params, use_tqdm=True)
    else:
        all_probe_outputs = []

    # --- ### PHASE 4: ASSEMBLE RESULTS AND SAVE ### ---
    print("\n--- Phase 4: Assembling results and saving confidence data ---")
    output_cursor = 0
    for sample_map in tqdm(results_map, desc="Assembling & Saving"):
        final_confidence_data_for_sample = []
        sample_idx = sample_map["sample_index"]
        
        for step_info in sample_map["thinking_steps"]:
            num_probes = step_info["num_probes"]
            
            probe_outputs_for_step = all_probe_outputs[output_cursor : output_cursor + num_probes]
            
            avg_confidence, all_top_k_met = calculate_metrics_from_outputs_multi(
                probe_outputs=probe_outputs_for_step,
                target_gt_token_ids=step_info["target_gt_token_ids"],
                top_k_factor=args.top_k_factor
            )
            
            final_confidence_data_for_sample.append({
                "token_index": step_info["token_index"],
                "token_id": step_info["token_id"],
                "decoded_token": step_info["decoded_token"],
                "confidence": avg_confidence,
                "all_top_k_met": all_top_k_met
            })
            output_cursor += num_probes

        # Save the assembled data for this one sample
        output_confidence_file = f"./step1_confidence_data/{args.dataset_name}/{args.model_path.split('/')[-1]}/{sample_idx}_{args.dataset_name}_confidence_data.jsonl"
        output_directory = os.path.dirname(output_confidence_file)
        os.makedirs(output_directory, exist_ok=True)
        
        with open(output_confidence_file, "w") as f:
            json.dump(final_confidence_data_for_sample, f, ensure_ascii=False, indent=4)
            
    print(f"\nProcessing complete. Save in {output_confidence_file}")


if __name__ == "__main__":
    main_example()