from utils.data_gen import *
import collections
import csv
from scipy.stats import chi2_contingency


selected_tokens = ['how', 'Wait', 'but','check','question','problem',
'So', 'Therefore', 'right', 'Then','?']
suspecting_tokens = ['how', 'Wait', 'but']
proving_tokens = ['So', 'Therefore', 'right', 'Then', '?']

def chi_analysis_from_aggregated_selected(csv_path):
    """
    对 gsm8k_AGGREGATED_SELECTED_tokens.csv
    直接进行 Peak / Valley × Token-Category 的卡方检验
    """
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    df.columns = df.columns.str.strip()
    df['Token'] = df['Token'].str.strip()

    # 2. 只保留我们关心的 token
    df = df[df['Token'].isin(suspecting_tokens + proving_tokens)]

    if df.empty:
        print("❌ 没有可用于统计的 token")
        return

    # 3. 映射类别
    def category(token):
        return 'Suspecting' if token in suspecting_tokens else 'Proving'

    df['Category'] = df['Token'].apply(category)

    # 4. 构造列联表
    contingency = (
        df.groupby('Category')[['Peak_Count', 'Valley_Count']]
        .sum()
        .T
    )
    contingency.index = ['Peak', 'Valley']

    print("\n=== Chi-square Contingency Table ===")
    print(contingency)

    # 5. 卡方检验
    chi2, p, dof, expected = chi2_contingency(contingency)

    print("\n=== Chi-square Test Result ===")
    print(f"Chi² = {chi2:.4f}")
    print(f"p-value = {p:.4e}")
    print(f"dof = {dof}")

    expected_df = pd.DataFrame(
        expected,
        index=contingency.index,
        columns=contingency.columns
    )
    print("\nExpected Frequencies:")
    print(expected_df.round(2))

    # 6. 解释（论文可直接用）
    if p < 0.05:
        print(
            "\n✅ Significant result: "
            "Token category is statistically associated with reasoning phase (peak vs valley)."
        )
    else:
        print(
            "\n❌ Not significant: "
            "No statistical evidence of association."
        )


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
    data = data['train']
    if len(data) > 1000:
        data = data.select(range(1000))
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
    # print(f"Attempting to use tensor_parallel_size: {args.tensor_parallel_size}, vLLM max_model_len: {vllm_max_model_len}")
    


    # 统计all_point对应token
    peak_token_counts = collections.Counter()
    valley_token_counts = collections.Counter()
    # 初始化峰和谷的总数计数器
    total_peak_points = 0
    total_valley_points = 0

    for i, (prompt_for_model, question, answer_text) in enumerate(tqdm(zip(prompts_for_model, list(qa_datas_subset.keys()), list(qa_datas_subset.values())), total=len(prompts_for_model), desc="Processing dataset")):

        correct_answer = dataset_handler.extract_answer(answer_text)

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
            
            pruned_think_text, confidence_data_for_graph, all_points = prune_by_rise_over_fall(
                    confidence_data=collected_token_data,
                    tokenizer=tokenizer,
                    window_size = args.window_size,
                    peak_prominence=args.peak_prominence,
            )

            
            window_radius = args.window_size // 2
            
            for point in all_points:
                point_index = point['index']
                point_type = point['type']
                
                # Define the slice of data around the turning point
                start_idx = max(0, point_index - window_radius)
                # +1 because Python slicing is exclusive at the end
                end_idx = min(len(confidence_data_for_graph), point_index + window_radius + 1)
                
                # Determine which counter to update
                if point_type == 'peak':
                    counter_to_update = peak_token_counts
                    total_peak_points += 1  # 峰总数加一
                elif point_type == 'valley':
                    counter_to_update = valley_token_counts
                    total_valley_points += 1 # 谷总数加一
                else:
                    continue # Skip if not a peak or valley

                # Iterate through the tokens in the window and count them
                for token_idx in range(start_idx, end_idx):
                    # Ensure the data exists at this index
                    if token_idx < len(confidence_data_for_graph):
                        token_id = confidence_data_for_graph[token_idx]['token_id']
                        # Decode the single token_id to its string representation
                        token_str = tokenizer.decode([token_id])
                        counter_to_update[token_str] += 1



        # 原始输出，剪枝前：generated_full_output_text
        # prompt输入：prompt_for_model
        # pruned_response_str
        # 剪枝后think：pruned_think_text

    # ==============================================================================
    # START: MODIFIED TO SAVE COMBINED RISE/FALL RESULTS WITH PANDAS
    # ==============================================================================


    print("\n\n" + "="*80)
    print("--- Combining and Saving Token Frequency Analysis with Pandas ---")
    print("="*80)

    peak_series = pd.Series(peak_token_counts, name='Peak_Count')
    valley_series = pd.Series(valley_token_counts, name='Valley_Count')
    combined_df = pd.concat([peak_series, valley_series], axis=1)
    combined_df.fillna(0, inplace=True)
    combined_df['Total_Count'] = combined_df['Peak_Count'] + combined_df['Valley_Count']
    combined_df.sort_values(by='Total_Count', ascending=False, inplace=True)
    combined_df = combined_df.astype(int)

    output_directory = f"./token_frequency_results/{args.model_path.split('/')[-1]}/"
    os.makedirs(output_directory, exist_ok=True)
    combined_csv_path = os.path.join(output_directory, f"{args.dataset_name}_combined_peak_valley_tokens.csv")

    try:
        # FIX 1: Add quoting=csv.QUOTE_ALL
        combined_df.to_csv(combined_csv_path, index_label='Token', quoting=csv.QUOTE_ALL)
        print(f"\nSuccessfully saved combined token frequencies to: {combined_csv_path}")
    except IOError as e:
        print(f"\nError saving combined token frequencies: {e}")

    print("\n" + "="*80)
    # ==============================================================================
    # END: MODIFIED TO SAVE COMBINED RESULTS WITH PANDAS
    # ==============================================================================

    try:
        print("\n\n" + "="*80)
        print("--- Combining and Saving Full Token Frequency Analysis ---")
        print("="*80)

        # Re-create the DataFrame for the full report to ensure correctness
        peak_series_full = pd.Series(peak_token_counts, name='Peak_Count')
        valley_series_full = pd.Series(valley_token_counts, name='Valley_Count')
        combined_df_full = pd.concat([peak_series_full, valley_series_full], axis=1)
        combined_df_full.fillna(0, inplace=True)
        combined_df_full['Total_Count'] = combined_df_full['Peak_Count'] + combined_df_full['Valley_Count']
        combined_df_full.sort_values(by='Total_Count', ascending=False, inplace=True)
        combined_df_full = combined_df_full.astype(int)

        full_csv_path = os.path.join(output_directory, f"{args.dataset_name}_FULL_peak_valley_tokens.csv")
        # FIX 2: Add quoting=csv.QUOTE_ALL
        combined_df_full.to_csv(full_csv_path, index_label='Token', quoting=csv.QUOTE_ALL)
        print(f"\nSuccessfully saved FULL token frequencies to: {full_csv_path}")

        # The rest of your analysis code...
        normalized_index = combined_df_full.index.str.strip()
        aggregated_df = combined_df_full.groupby(by=normalized_index).sum()
        final_filtered_df = aggregated_df[aggregated_df.index.isin(selected_tokens)].copy() # Use .copy() to avoid warnings

        if not final_filtered_df.empty:
            # Check for division by zero before calculating probabilities
            if total_peak_points > 0:
                final_filtered_df['Rise_prob'] = final_filtered_df['Peak_Count'] / total_peak_points
            else:
                final_filtered_df['Rise_prob'] = 0

            if total_valley_points > 0:
                final_filtered_df['Fall_prob'] = final_filtered_df['Valley_Count'] / total_valley_points
            else:
                final_filtered_df['Fall_prob'] = 0

            final_filtered_df['Difference (Peak - Valley)'] = final_filtered_df['Rise_prob'] - final_filtered_df['Fall_prob']
            
            # Note: The original 'Diff Rate' calculation was based on the probability difference,
            # but the denominator used raw counts. This might be unintentional.
            # Using the raw count difference for the rate calculation for consistency.
            raw_diff = final_filtered_df['Peak_Count'] - final_filtered_df['Valley_Count']
            denominator = final_filtered_df[['Peak_Count', 'Valley_Count']].min(axis=1)
            final_filtered_df['Diff Rate'] = raw_diff.divide(denominator.replace(0, pd.NA)).fillna(0)

            final_filtered_df = final_filtered_df.sort_values(by='Difference (Peak - Valley)', ascending=False)
            
            filtered_csv_path = os.path.join(output_directory, f"{args.dataset_name}_AGGREGATED_SELECTED_tokens.csv")
            
            # FIX 3: Add quoting=csv.QUOTE_ALL
            final_filtered_df.to_csv(filtered_csv_path, index_label='Token', quoting=csv.QUOTE_ALL)
            print(f"Successfully saved AGGREGATED and filtered token analysis to: {filtered_csv_path}")
            print("\nFinal Aggregated & Filtered Data:")
            print(final_filtered_df)
        else:
            print("None of the selected tokens were found in the results after aggregation.")

    except (IOError, KeyError) as e:
        print(f"\nError processing or saving token frequencies: {e}")

   
    chi_analysis_from_aggregated_selected(filtered_csv_path)


if __name__ == "__main__":
    main_example()