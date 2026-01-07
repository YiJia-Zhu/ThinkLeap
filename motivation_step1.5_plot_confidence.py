from utils.data_gen import *
import csv
import re


def main_example():
    args = parse_arguments_for_example()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
        
    tensor_parallel_size = len(args.cuda_visible_devices.split(','))
    # sample_data ={"question": "Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?",
    # "answer": "18"}
    # problem_text = sample_data["question"]
    # ground_truth_final_answer_for_pruning = sample_data["answer"]
    # # Prepare messages for Qwen's chat template
    # messages = [{"role": "user", "content": problem_text}]
    
    # prompt_for_model = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True, # Adds the role for the assistant to start generating
    #     enable_thinking=True,
    # )
    print(f"Loading tokenizer from: {args.model_path}")
    try:
        # For Qwen models, trust_remote_code=True is often necessary for the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return

    dataset_handler = get_dataset(args.dataset_name)
    data, answer_type = dataset_handler.load_data()


    if "math500" in args.dataset_name:
        # we split 300 for training 200 for eval
        data = data['test']
        if len(data) > 300: data = data.select(range(300))
        args.tokenizer_default_max_length = 10240 + 4096
        args.max_new_tokens = 10240 +2048
    elif "gsm8k" in args.dataset_name:
        data = data['train']
        if len(data) > 360: data = data.select(range(360))
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
    # Split data into subsets
    total_questions = len(qa_data)
    subset_size = total_questions // args.num_subsets
    start_index = args.subset_index * subset_size
    end_index = total_questions if args.subset_index == args.num_subsets - 1 else (args.subset_index + 1) * subset_size
    qa_data_subset = {k: qa_data[k] for k in list(qa_data.keys())[start_index:end_index]}
    
    # Generate prompts with tokenizer
    prompts_for_model, qa_datas_subset = generate_prompt(args, logger, qa_data_subset, answer_type=answer_type, tokenizer=tokenizer)
    # print("-----------------")
    # print(list(qa_data_subset.values())[0])
    # print("-----------------")

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

    text_qwen_think_end_token = tokenizer.decode([THINK_END_ID], skip_special_tokens=False) # e.g., "</think>"


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
    
    # vllm_engine = LLM(
    #     model=args.model_path,
    #     tokenizer=args.model_path,
    #     tensor_parallel_size=tensor_parallel_size,
    #     max_model_len=vllm_max_model_len, 
    #     trust_remote_code=True, # Often needed for Qwen
    #     gpu_memory_utilization=args.gpu_memory_utilization,
    #     # enable_chunked_prefill=True,
    #     dtype="bfloat16",
    #     seed=1,
    # )
    # Actual max model length used by vLLM engine
    # effective_max_model_len = vllm_engine.llm_engine.model_config.max_model_len
    # # print(f"vLLM Engine initialized. Effective Max model length: {effective_max_model_len}. Using {tensor_parallel_size} GPU(s).")

    # gen_sampling_params = SamplingParams(
    #     max_tokens=args.max_new_tokens,
    #     temperature=args.temperature if args.do_sample else 0.0,
    #     top_p=args.top_p if args.do_sample else 1.0,
    #     stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [],
        
    #     # We want the model to generate its full thought and answer.
    # )

    
    for i, (prompt_for_model, question, answer_text) in enumerate(tqdm(zip(prompts_for_model, list(qa_datas_subset.keys()), list(qa_datas_subset.values())), total=len(prompts_for_model), desc="Processing dataset")):
        correct_answer = dataset_handler.extract_answer(answer_text)
        if i<2:
            continue
        org_think_text = ""
        org_think_sequence_ids = []
        collected_token_data = None
        
        input_confidence_file = f"./step1_confidence_data/{args.dataset_name}/{args.model_path.split('/')[-1]}/{i}_{args.dataset_name}_confidence_data.jsonl"
        
        if not os.path.exists(input_confidence_file):
            print(f"Warning: Pre-computed confidence file not found, skipping item {i}. Path: {input_confidence_file}")
            continue

        with open(input_confidence_file, "r", encoding='utf-8') as f:
            collected_token_data = json.load(f)

        if collected_token_data and len(collected_token_data) > 1:
            # The first item in the list is a placeholder, so we skip it ([1:]).
            org_think_sequence_ids = [token_data['token_id'] for token_data in collected_token_data[1:]]
            org_think_text = tokenizer.decode(org_think_sequence_ids)
        else:
            print(f"Warning: Confidence data for item {i} is empty or invalid.")
            continue


        if not org_think_text.strip():
            print("Warning: Content within Qwen think block (before end token) is empty or whitespace. Skipping pruning of this block.")
            # pruned_response_str remains as original prompt + raw vLLM output
        else:

            
            # 步骤 2: 调用第二个函数，根据收集到的数据进行剪枝
            # 这一步很快，不与 vLLM 交互，只应用剪枝逻辑
            
            pruned_think_text, confidence_data_for_graph,unique_points = prune_by_rise_over_fall(
                    confidence_data=collected_token_data,
                    tokenizer=tokenizer,
                    window_size = args.window_size,
                    peak_prominence=args.peak_prominence,
                    rise_magnitude = args.rise_magnitude,
                    only_early_stop = args.only_early_stop,
                    is_skeptical = args.is_skeptical
            )
            # plot_pruning_confidence_12(confidence_data_for_graph,unique_points=unique_points, output_image_path="./plot_fig/"+str(i)+".pdf", window_size = args.window_size,)
            plot_pruning_confidence_12(confidence_data_for_graph, output_image_path="./plot_fig/"+str(i)+".pdf", window_size = args.window_size,)

            # print("-----------------------")
            # print(pruned_think_text)
            # exit()
            # if i>12:
            break

            # text_after_think_segment_generated = tokenizer.decode(ids_after_think_block_generated, skip_special_tokens=False)

            org_response_str = (
                prompt_for_model +
                org_think_text +
                text_qwen_think_end_token
            )
            pruned_response_str = (
                prompt_for_model +
                pruned_think_text +
                text_qwen_think_end_token
            )
            batched_outputs = vllm_engine.generate([org_response_str, pruned_response_str], gen_sampling_params, use_tqdm=False)

            if len(batched_outputs) != 2:
                print(f"Error: Batch inference did not return the expected number of outputs for item {i}. Skipping.")
                continue

            # Process the output from the original (unpruned) thinking text
            org_full_output_text = batched_outputs[0].outputs[0].text
            org_is_correct = dataset_handler.check(correct_answer, dataset_handler.extract_answer(org_full_output_text))
            
            # Process the output from the pruned thinking text
            pruned_outputs_str = batched_outputs[1].outputs[0].text
            pruned_is_correct = dataset_handler.check(correct_answer, dataset_handler.extract_answer(pruned_outputs_str))

        
            data_to_save = {
                "question": question,
                "correct_answer_text": answer_text,
                "correct_answer": correct_answer,
                "org_think_text": org_think_text,
                "org_is_correct": org_is_correct,
                'pruned_think_text': pruned_think_text,
                "pruned_is_correct": pruned_is_correct,
            }

            # 追加写入 JSONL 文件
            # output_jsonl_file = f"./step1_compressed_data/{args.dataset_name}/{args.model_path.split('/')[-1]}/{args.dataset_name}_{args.window_size}_{args.peak_prominence}_{args.rise_magnitude}.jsonl"
            output_directory = os.path.dirname(args.output_jsonl_file)
            # 创建目录（如果它还不存在的话）
            # exist_ok=True 表示如果目录已存在，则不会引发错误
            if not os.path.exists(output_directory):
                os.makedirs(output_directory, exist_ok=True)
                print(f"Created directory: {output_directory}")
            with open(args.output_jsonl_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(data_to_save, ensure_ascii=False) + '\n')


        # 原始输出，剪枝前：generated_full_output_text
        # prompt输入：prompt_for_model
        # pruned_response_str
        # 剪枝后think：pruned_think_text




if __name__ == "__main__":

    main_example()