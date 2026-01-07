import os
import json
import csv
import argparse
from pathlib import Path

def analyze_json_file(file_path: str):
    """
    Analyzes a JSON file to calculate accuracy and average token consumption.

    Args:
        file_path (str): The path to the JSON results file.

    Returns:
        tuple: A tuple containing (accuracy, avg_token_consumption).
               Returns (None, None) if the file cannot be processed.
    """
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Warning: File not found '{file_path}'. Skipping.")
        return None, None

    # Read the JSON file
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    except json.JSONDecodeError:
        print(f"Warning: File '{file_path}' is not a valid JSON file. Skipping.")
        return None, None
    except Exception as e:
        print(f"Warning: An error occurred while reading '{file_path}': {e}. Skipping.")
        return None, None

    # Check if the data is a non-empty list
    if not results_data or not isinstance(results_data, list):
        print(f"Warning: JSON data in '{file_path}' is empty or not a list. Skipping.")
        return None, None

    # Initialize metrics
    correct_predictions = 0
    total_tokens_consumed = 0
    total_samples = len(results_data)

    # Process each entry in the data
    for entry in results_data:
        if entry.get("is_correct", False):
            correct_predictions += 1
        total_tokens_consumed += entry.get("tokens_generated", 0)

    # Calculate final results
    if total_samples > 0:
        accuracy = (correct_predictions / total_samples) * 100
        avg_token_consumption = total_tokens_consumed / total_samples
        return accuracy, avg_token_consumption
    else:
        print(f"Warning: No sample data found in '{file_path}'. Skipping.")
        return None, None

def parse_path_info(file_path: str):
    """
    Extracts method, model, dataset, and epoch from the given file path.

    Example Paths:
    ./step4_eval_data/main_exp/Qwen3-14B/arc_challenge_10_0.03_0.01_epoch1.json
    ./step4_eval_data/main_exp/Qwen3-14B/math500_org.json

    Args:
        file_path (str): The full path to the results file.

    Returns:
        tuple: A tuple containing (method, model, dataset, epoch).
               Returns (None, None, None, None) if the path format is unexpected.
    """
    try:
        p = Path(file_path)
        filename = p.name
        
        # Extract model name (parent directory name)
        model = p.parent.name
        
        method = None
        dataset = None
        epoch = None

        if filename.endswith('_org.json'):
            method = 'org'
            dataset = filename.removesuffix('_org.json')
            epoch = -1  # As requested for 'org' method
            
        elif '_epoch' in filename and filename.endswith('.json'):
            method = 'ours'
            # Split the filename at the last occurrence of '_epoch'
            base_name, epoch_str = filename.removesuffix('.json').rsplit('_epoch', 1)
            dataset = base_name
            epoch = int(epoch_str)
        else:
            # print(f"Warning: Filename format not recognized for epoch parsing '{filename}'. Skipping.")
            method = 'general'
            dataset = filename.split('_')[0]
            epoch = -1  # As requested for 'org' method
            # return None, None, None, None
            # 
        return method, model, dataset, epoch
        
    except (IndexError, ValueError) as e:
        print(f"Warning: Could not parse info from filename '{p.name}': {e}. Skipping.")
        return None, None, None, None
    except Exception as e:
        print(f"Warning: Could not parse path '{file_path}': {e}. Skipping.")
        return None, None, None, None

def main(root_dir: str, output_csv: str):
    """
    Main function to scan directories, analyze files, and compile a CSV report.
    """
    all_results = []
    
    print(f"--- Starting Analysis ---")
    print(f"Scanning for JSON files in: {root_dir}")

    # Walk through the directory structure
    for dirpath, _, filenames in os.walk(root_dir):
        # We are only interested in files within a 'main_exp' directory
        if 'main_exp' not in dirpath:
            continue

        for filename in filenames:
            if filename.endswith('.json'):
                full_path = os.path.join(dirpath, filename)


                if "10_0.03_0.01" in full_path:
                    continue



                    
                print(f"Processing file: {full_path}")


                # 1. Parse metadata from the path
                method, model, dataset, epoch = parse_path_info(full_path)
                if not all((method, model, dataset)) or epoch is None:
                    continue

                # 2. Analyze the JSON file content
                accuracy, avg_tokens = analyze_json_file(full_path)
                if accuracy is None or avg_tokens is None:
                    continue
                
                # 3. Store the result
                all_results.append({
                    'method': method,
                    'model': model,
                    'dataset': dataset,
                    'epoch': epoch,  # Add the new epoch field
                    'accuracy': f"{accuracy:.2f}",
                    'avg_token_consumption': f"{avg_tokens:.2f}"
                })

    if not all_results:
        print("--- Analysis Complete ---")
        print("No valid result files were found to compile.")
        return

    # 4. Write all collected data to a CSV file
    print(f"\n--- Writing Results ---")
    print(f"Found {len(all_results)} valid results. Saving to {output_csv}")
    
    # Sort results for consistent output, now including epoch
    all_results.sort(key=lambda x: (x['method'], x['model'], x['dataset'], x['epoch']))

    try:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            # Add 'epoch' to the CSV header
            fieldnames = ['method', 'model', 'dataset', 'epoch', 'accuracy', 'avg_token_consumption']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        print("Successfully created the CSV report.")
    except Exception as e:
        print(f"Error: Failed to write to CSV file '{output_csv}': {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze model inference results from a nested directory structure and compile them into a single CSV file."
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        default="./step5_eval_data/main_exp",
        help="The root directory to start scanning for JSON files. It should contain model subdirectories."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./main_exp.csv",
        help="The path to the output CSV file."
    )
    
    args = parser.parse_args()
    
    main(args.root_dir, args.output_csv)