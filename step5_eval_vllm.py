import argparse
import json
import os
import sys
from tqdm import tqdm
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from utils.dataset_loader import *
from typing import List
# MODIFIED: Import LoRARequest
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
# Initialize logger
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

seed = 1
torch.manual_seed(seed)
np.random.seed(seed)

def parse_arguments_for_example():
    """Parses command-line arguments for the example script."""
    parser = argparse.ArgumentParser(description="Run model generation from a local JSON file with LoRA weights.")
    
    # --- Batching & Output Arguments ---
    parser.add_argument(
        "--batch_size", type=int, default=10240,
        help="Batch size for vLLM inference. No use in fact"
    )
    parser.add_argument(
        "--output_file", type=str, default="./step5_eval_data/ablation/org_inference_results_lora.json",
        help="Path to save the inference results in JSON format."
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="0",
        help="Comma-separated list of GPU IDs to use (e.g., '0,1'). This determines the tensor_parallel_size."
    )
    
    # --- Model and Dataset Arguments ---
    parser.add_argument(
        "--dataset_name", type=str, default = "gsm8k",
        help="Name of the dataset handler to use for answer extraction (e.g., gsm8k).")
    # parser.add_argument(
    #     "--input_json_path", type=str, 
    #     default="./step3_training_data/gsm8k_full_test_data.json",
    #     help="Path to the local input JSON file."
    # )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./huggingface_models/QwQ-32B",
        help="Path to the BASE Hugging Face model."
    )
    
    # --- NEW: LoRA Specific Arguments ---
    parser.add_argument(
        "--lora_path",
        type=str,
        default=None,
        help="Path to the directory containing the LoRA weights (adapter_model.bin)."
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="The rank of the LoRA adapter. Should match the rank used during training."
    )
    
    # --- Other Arguments ---
    parser.add_argument(
        "--max_new_tokens", type=int, default=4096,
        help="Maximum new tokens for initial solution generation."
    )
    parser.add_argument(
        "--temperature", type=float, default=0,
        help="Temperature for sampling."
    )
    parser.add_argument(
        "--top_p", type=float, default=1,
        help="Top-p for sampling."
    )
    parser.add_argument(
        "--gpu_memory_utilization", type=float, default=0.9,
        help="GPU memory utilization factor for vLLM (0.0 to 1.0)."
    )
    parser.add_argument(
        "--tokenizer_default_max_length", type=int, default=8192,
        help="Default model_max_length to set for the tokenizer if not set or too large."
    )
    parser.add_argument(
        "--tokenizer_max_length_override_threshold", type=int, default=20480,
        help="Threshold above which tokenizer's model_max_length will be overridden."
    )

    return parser.parse_args()


def main_example():
    args = parse_arguments_for_example()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
        
    tensor_parallel_size = len(args.cuda_visible_devices.split(','))


    if "gsm8k" in args.dataset_name:
        input_json_path = "./step3_training_data/gsm8k_full_test_data.json"
    elif "math500" in args.dataset_name:
        input_json_path = "./step3_training_data/math500_full_test_data.json"
        args.tokenizer_default_max_length = 10240 + 4096
        args.max_new_tokens =10240 +2048
    elif "aime24" in args.dataset_name:
        input_json_path = "./step3_training_data/aime24_full_test_data.json"
        args.tokenizer_default_max_length = 10240 + 4096
        args.max_new_tokens = 10240 +2048
    elif "amc23" in args.dataset_name:
        input_json_path = "./step3_training_data/amc23_full_test_data.json"
        args.tokenizer_default_max_length = 10240 + 4096
        args.max_new_tokens = 10240 +2048
    elif "arc_challenge" in args.dataset_name:
        input_json_path = "./step3_training_data/arc_challenge_full_test_data.json"
        args.tokenizer_default_max_length = 10240 + 4096
        args.max_new_tokens = 10240 +2048
    
    print(f"Loading tokenizer from: {args.model_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    # --- Data Loading (remains the same) ---
    print(f"Loading data from local JSON file: {input_json_path}")
    dataset_handler = get_dataset(args.dataset_name)
    questions = []
    ground_truth_answers = []
    try:
        with open(input_json_path, 'r', encoding='utf-8') as f:
            local_data = json.load(f)
            prompt_text = local_data[0].get("system","")

        for item in local_data:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            output_text = item.get("output", "")
            question = f"{instruction}\n{input_text}".strip()
            questions.append(question)
            ground_truth_answers.append(output_text)
    except FileNotFoundError:
        print(f"Error: The file {input_json_path} was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from the file {input_json_path}.")
        return
    print(f"Loaded {len(questions)} samples from the JSON file.")

    prompts_for_model = []
    for q in questions:
        messages = [
                {"role": "system", "content": prompt_text},
                {"role": "user", "content": f"Question: {q}"},
                ]
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        prompts_for_model.append(prompt)

    # --- Tokenizer Settings (remains the same) ---
    current_max_len = getattr(tokenizer, 'model_max_length', None)
    if current_max_len is None or current_max_len > args.tokenizer_max_length_override_threshold:
        tokenizer.model_max_length = args.tokenizer_default_max_length
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- MODIFIED: Initialize vLLM Engine with LoRA Support ---
    print(f"\n--- Initializing vLLM Engine from: {args.model_path} ---")
    if args.lora_path:
        print(f" >> LoRA support ENABLED. Loading adapter from: {args.lora_path}")
    vllm_max_model_len = int(tokenizer.model_max_length)
    
    vllm_engine = LLM(
        model=args.model_path,
        tokenizer=args.model_path,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=vllm_max_model_len, 
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype="bfloat16",
        # NEW: Enable LoRA and specify max rank and number of adapters
        enable_lora=True if args.lora_path else False,
        seed=1,
        max_loras=1,
        max_lora_rank=args.lora_rank
    )
    print(f"vLLM Engine initialized. Effective Max model length: {vllm_engine.llm_engine.model_config.max_model_len}. Using {tensor_parallel_size} GPU(s).")
    
    gen_sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [],
    )
    
    # --- NEW: Define the LoRA Request ---
    lora_request = None
    if args.lora_path:
        # The first argument is an arbitrary name you give to the adapter.
        # The second argument is the path to the adapter weights.
        lora_request = LoRARequest("my_custom_lora", 1, args.lora_path)

    # --- BATCHED INFERENCE AND EVALUATION ---
    # ... (the rest of the loop logic is the same)
    correct_predictions = 0
    total_samples_processed = 0
    total_tokens_consumed = 0
    all_results_data = []

    for i in range(0, len(prompts_for_model), args.batch_size):
        batch_prompts = prompts_for_model[i:i + args.batch_size]
        batch_questions = questions[i:i + args.batch_size]
        batch_ground_truth = ground_truth_answers[i:i + args.batch_size]
        
        # MODIFIED: Pass the lora_request to the generate method
        vllm_outputs = vllm_engine.generate(
            batch_prompts, 
            gen_sampling_params, 
            use_tqdm=True,
            lora_request=lora_request # Pass the LoRA request here
        )
        
        for j, output in enumerate(vllm_outputs):
            if not output.outputs:
                print(f"Warning: vLLM returned no output for prompt index {i+j}.")
                continue

            generated_full_output_text = output.outputs[0].text
            tokens_generated = len(output.outputs[0].token_ids)
            total_tokens_consumed += tokens_generated
            
            correct_answer_text = batch_ground_truth[j]
            extracted_ground_truth = dataset_handler.extract_answer(correct_answer_text)
            extracted_model_answer = dataset_handler.extract_answer(generated_full_output_text)
            
            is_correct = dataset_handler.check(extracted_ground_truth, extracted_model_answer)
            
            if is_correct:
                correct_predictions += 1
            
            total_samples_processed += 1
            
            result_entry = {
                "index": i + j,
                "question": batch_questions[j],
                "ground_truth_full": correct_answer_text,
                "ground_truth_extracted": extracted_ground_truth,
                "model_output": generated_full_output_text,
                "model_answer_extracted": extracted_model_answer,
                "is_correct": is_correct,
                "tokens_generated": tokens_generated
            }
            all_results_data.append(result_entry)
            
    output_directory = os.path.dirname(args.output_file)
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)
        print(f"Created directory: {output_directory}")
    # --- Save results and report metrics (no changes here) ---
    print(f"\nSaving inference results to {args.output_file}...")
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            json.dump(all_results_data, f, ensure_ascii=False, indent=4)
        print("Successfully saved results.")
    except Exception as e:
        print(f"Error saving results to file: {e}")

    if total_samples_processed > 0:
        accuracy = (correct_predictions / total_samples_processed) * 100
        avg_token_consumption = total_tokens_consumed / total_samples_processed
        
        print("\n--- Evaluation Complete ---")
        print(f"Total Samples Processed: {total_samples_processed}")
        print(f"Correct Predictions: {correct_predictions}")
        print(f"Accuracy (ACC): {accuracy:.2f}%")
        print(f"Total Tokens Consumed: {total_tokens_consumed}")
        print(f"Average Token Consumption per Sample: {avg_token_consumption:.2f}")
    else:
        print("\nNo samples were processed.")


if __name__ == "__main__":
    main_example()
