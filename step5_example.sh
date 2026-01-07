python step5_eval_vllm.py
python step5_eval_vllm.py --lora_path ./saves/QwQ-32B-Instruct/lora/correct_train_2025-06-19-19-23-58/checkpoint-29 --output_file ./step5_eval_data/correct_inference_results_lora.json
python step5_eval_vllm.py --lora_path ./saves/QwQ-32B-Instruct/lora/train_2025-06-20-13-07-23/checkpoint-250 --output_file ./step5_eval_data/our_lora_inference_results_lora.json


python step5_eval_vllm.py --lora_path ./saves/QwQ-32B-Instruct/lora/train_2025-06-20-15-56-26/checkpoint-300 --output_file ./step5_eval_data/correct_inference_results_lora.json

python step5_eval_vllm.py --lora_path ./saves/QwQ-32B-Instruct/lora/no_skep_train_2025-06-20-18-10-54/checkpoint-100 --output_file ./step5_eval_data/no_skep_inference_results_lora.json


python step5_eval_vllm.py --lora_path ./saves/QwQ-32B-Instruct/ablation/gsm8k_our --output_file ./step5_eval_data/our_622.json --dataset_name gsm8k --cuda_visible_devices "1,2"


python step5.5_eval_from_json.py --input_file ./step5_eval_data/org_inference_results_lora.json


python step5.5_eval_from_json.py --input_file ./step5_eval_data/QwQ-32B/gsm8k_10_0.03_0.01_train_data.json

