import json
import os
import argparse
from tqdm import tqdm
import sys
from transformers import AutoTokenizer
from utils.dataset_loader import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

THINK_TOKEN_STR = "<think>"
THINK_END_TOKEN_STR = "</think>"


def convert_to_llama_factory_format(
    args,
    model_path,
    dataset_name,
    subset_index=0,
    num_subsets=1
):
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return
    output_file = (
        f"./step3_training_data/{dataset_name}_full_test_data.json"
    )



    dataset_handler = get_dataset(dataset_name)
    data, answer_type = dataset_handler.load_data()
    PROMPT = get_prompt(answer_type, args.non_reasoning_model)

    # =====================================================
    # Select TEST data only (dataset-specific rules)
    # =====================================================
    if "gsm8k" in dataset_name:
        data = data["test"]
        if len(data) > 10000:
            data = data.select(range(10000))
        test_size = int(len(data) * 0.2)
        data = data.select(range(test_size))

    elif "math500" in dataset_name:
        # Dataset doesn't have train, test split
        data = data["test"]
        if len(data) > 10000:
            data = data.select(range(10000))
        test_size = int(len(data) * 0.4) 
        data = data.select(range(len(data) - test_size, len(data)))

    elif "aime24" in dataset_name:
        # Dataset too samll, all for eval
        data = data["train"]
        if len(data) > 100:
            data = data.select(range(100))

    elif "arc_challenge" in dataset_name:
        data = data["test"]
        if len(data) > 300:
            data = data.select(range(300))

    elif "amc23" in dataset_name:
        data = data["test"]
        if len(data) > 100:
            data = data.select(range(100))

    else:
        data = data["train"]
        if len(data) > 500:
            data = data.select(range(500))

    # =====================================================
    # Prepare QA data
    # =====================================================
    qa_data = dataset_handler.prepare_qa_data(data)

    # Subset slicing (for multi-GPU / multi-job usage)
    total_questions = len(qa_data)
    subset_size = total_questions // num_subsets
    start_index = subset_index * subset_size
    end_index = (
        total_questions
        if subset_index == num_subsets - 1
        else (subset_index + 1) * subset_size
    )
    qa_data_subset = {
        k: qa_data[k]
        for k in list(qa_data.keys())[start_index:end_index]
    }

    # Generate prompts
    _, qa_datas_subset = generate_prompt(
        args,
        logger,
        qa_data_subset,
        answer_type=answer_type,
        tokenizer=tokenizer,
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    test_samples = []
    lines = 0

    # =====================================================
    # Build LLaMA-Factory TEST samples
    # =====================================================
    for question, answer_text in tqdm(
        qa_datas_subset.items(),
        total=len(qa_datas_subset),
        desc="Processing dataset",
    ):
        if "gsm8k" in dataset_name or "aime24" in dataset_name:
            output = (
                f"{THINK_TOKEN_STR}\n{answer_text}"
                f"{THINK_END_TOKEN_STR}\n\n"
                f"Answer: {dataset_handler.extract_answer(answer_text)}"
            )
        elif "math500" in dataset_name:
            output = (
                f"{THINK_TOKEN_STR}\n{answer_text}"
                f"{THINK_END_TOKEN_STR}\n"
                f" \\boxed{{{dataset_handler.extract_answer(answer_text)}}}"
            )
        else:
            output = (
                f"{THINK_TOKEN_STR}\n{answer_text}"
                f"{THINK_END_TOKEN_STR}\n\n"
                f"Answer: {dataset_handler.extract_answer(answer_text)}"
            )

        test_samples.append(
            {
                "instruction": question,
                "input": "",
                "output": output,
                "system": PROMPT,
            }
        )
        lines += 1


    # random.shuffle(converted_samples)

    # =====================================================
    # Write TEST file only
    # =====================================================
    print(f"Writing {len(test_samples)} TEST samples to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test_samples, f, ensure_ascii=False, indent=4)

    print("\n--- Conversion Summary ---")
    print(f"  Total TEST samples saved: {len(test_samples)}")
    print("--------------------------\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert dataset to LLaMA-Factory TEST-only format"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="all",
        help="all (5 datasets), gsm8k, math500, aime24, arc_challenge, amc23",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./huggingface_models/QwQ-32B",
        help="Path to Hugging Face model",
    )
    parser.add_argument(
        "--non_reasoning_model",
        type=bool,
        default=False,
        help="Disable reasoning prompt",
    )

    args = parser.parse_args()


    if args.dataset_name == "all":
        for d in ["gsm8k", "math500", "aime24", "arc_challenge", "amc23"]:
            convert_to_llama_factory_format(
                args,
                args.model_path,
                d
            )

    else:
        convert_to_llama_factory_format(
            args,
            args.model_path,
            args.dataset_name
        )
