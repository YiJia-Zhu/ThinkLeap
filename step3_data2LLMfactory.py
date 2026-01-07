import json
import os
import argparse
from tqdm import tqdm
from utils.dataset_loader import get_prompt
# import random # Imported for shuffling
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
THINK_TOKEN_STR = "<think>"
THINK_END_TOKEN_STR = "</think>"

def update_dataset_info(output_file_path):
    """
    Checks for dataset_info.json in the output directory.
    If it exists, adds or updates the entry for the new dataset.
    If it doesn't exist, creates the file and adds the first entry.
    """
    output_dir = os.path.dirname(output_file_path)
    if not output_dir:
        output_dir = "." # Handle case where output is in the current directory

    info_file_path = os.path.join(output_dir, "dataset_info.json")
    
    # Load existing data or initialize a new dictionary
    if os.path.exists(info_file_path):
        try:
            with open(info_file_path, 'r', encoding='utf-8') as f:
                dataset_info = json.load(f)
        except json.JSONDecodeError:
            logger.warning(f"Could not decode {info_file_path}. A new file will be created.")
            dataset_info = {}
    else:
        dataset_info = {}

    # Create the new entry
    file_name = os.path.basename(output_file_path)
    # Generate dataset alias from filename by removing .json extension
    dataset_alias = os.path.splitext(file_name)[0] 
    
    dataset_info[dataset_alias] = {
        "file_name": file_name,
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system"
        }
    }
    
    # Write the updated data back to the file
    logger.info(f"Updating {info_file_path} with entry for '{dataset_alias}'...")
    with open(info_file_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=4)

def convert_to_llama_factory_format(input_file, output_file, test_split_ratio, answer_type="number",non_reasoning_model=0):
    
    if "gsm8k" in input_file:
        answer_type = "number"
    elif "math500" in input_file:
        answer_type = "latex_compression"
    elif "aime24" in input_file:
        answer_type = "number"

    PROMPT = get_prompt(answer_type, non_reasoning_model)
    # Ensure the output directory exists
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    converted_samples = []
    skipped_count = 0



    # Reading logic remains the same (expecting JSONL input)
    with open(input_file, 'r', encoding='utf-8') as infile:
        lines = infile.readlines()
        for line in tqdm(lines, desc="Processing samples"):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed line: {line.strip()}")
                continue

            # Filter for high-quality samples where both original and pruned are correct
            if data.get("org_is_correct") and data.get("pruned_is_correct"):
                correct_answer = data.get("correct_answer", "")
                if not isinstance(correct_answer, str):
                    correct_answer = str(correct_answer)
                
                think_text = data.get('pruned_think_text', '')
                # The 'Explanation:' part is now implicitly part of the thinking process
                if think_text.startswith(THINK_TOKEN_STR):

                    # 如果是，则直接使用 think_text，不再添加开头的 token 和换行符
                    if answer_type == "latex_compression":
                        output = f"{think_text}{THINK_END_TOKEN_STR}\n\n"+"\\boxed{"+ correct_answer + "}"
                    else:
                        output = f"{think_text}{THINK_END_TOKEN_STR}\n\nAnswer: {correct_answer}"


                else:
                    if answer_type == "latex_compression":
                        output = f"{THINK_TOKEN_STR}\n{think_text}{THINK_END_TOKEN_STR}\n\n"+"\\boxed{"+ correct_answer + "}"
                    else:
                        # 如果不是（原来的逻辑），则在前面加上 token 和换行符
                        output = f"{THINK_TOKEN_STR}\n{think_text}{THINK_END_TOKEN_STR}\n\nAnswer: {correct_answer}"


                # Build the LLaMA-Factory format sample
                new_sample = {
                    "instruction": data.get("question", ""),
                    "input": "",
                    "output": output,
                    "system": PROMPT
                }
                converted_samples.append(new_sample)
            else:
                skipped_count += 1


    # Shuffle the data before splitting for a random distribution
    # random.shuffle(converted_samples)

    # Determine split index based on the ratio
    split_index = int(len(converted_samples) * test_split_ratio)
    
    test_samples = []
    train_samples = converted_samples

    if split_index > 0:
        print(f"\nSplitting data: {test_split_ratio*100:.1f}% for test set...")
        test_samples = converted_samples[:split_index]
        train_samples = converted_samples[split_index:]

        # Generate the test file path from the output file path
        output_dir = os.path.dirname(output_file)
        output_filename = os.path.basename(output_file)
        test_filename = output_filename.replace("train", "test")
        test_output_file = os.path.join(output_dir, test_filename)

        # Write the test data
        print(f"Writing {len(test_samples)} samples to {test_output_file}...")
        with open(test_output_file, 'w', encoding='utf-8') as outfile:
            json.dump(test_samples, outfile, ensure_ascii=False, indent=4)

        update_dataset_info(test_output_file)
    
    # Write the training data
    print(f"Writing {len(train_samples)} samples to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(train_samples, outfile, ensure_ascii=False, indent=4)

    update_dataset_info(output_file)

    print(f"\n--- Conversion Summary ---")
    print(f"  Total samples processed: {len(lines)}")
    print(f"  Low-quality samples skipped: {skipped_count}")
    print(f"  High-quality samples saved (Train): {len(train_samples)}")
    if test_samples:
        print(f"  High-quality samples saved (Test): {len(test_samples)}")
    print(f"--------------------------\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert custom JSONL data to LLaMA-Factory SFT format (JSON) and optionally create a test split.")
    
    parser.add_argument(
        "--input_file",
        type=str,
        default="./step2_compressed_data/gsm8k/QwQ-32B/ablation/gsm8k_10_0.03_0.01_pruned_data.jsonl",
        help="Path to the source JSONL file (e.g., gsm8k_pruned_data.jsonl).",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="./step3_training_data/gsm8k/QwQ-32B/gsm8k_10_0.03_0.01_pruned_data.json",
        help="Path to save the converted LLaMA-Factory training file (as .json).",
    )
    # Add the new argument for the test set split ratio
    parser.add_argument(
        "--test_split_ratio",
        type=float,
        default=0, # Default to a 10% test split
        help="The proportion of the data to be used as a test set (e.g., 0.1 for 10%). Set to 0 to disable.",
    )
    parser.add_argument(
        "--non_reasoning_model", type=bool, default=False,
        help="non_reasoning_model")

    args = parser.parse_args()

    # Validate test_split_ratio
    if not 0.0 <= args.test_split_ratio < 1.0:
        raise ValueError("test_split_ratio must be between 0.0 and 1.0 (exclusive of 1.0)")
    
    convert_to_llama_factory_format(args.input_file, args.output_file, args.test_split_ratio,args.non_reasoning_model)