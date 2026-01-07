import json
import argparse
import os

def analyze_inference_results(file_path: str):
    """
    Analyzes a JSON file containing inference results to calculate accuracy and token usage.

    Args:
        file_path (str): The path to the JSON results file.
    """
    # 检查文件是否存在
    if not os.path.exists(file_path):
        print(f"错误：找不到文件 '{file_path}'")
        return

    # 读取JSON文件
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            results_data = json.load(f)
    except json.JSONDecodeError:
        print(f"错误：文件 '{file_path}' 不是一个有效的JSON文件。")
        return
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        return

    # 检查数据是否为空
    if not results_data or not isinstance(results_data, list):
        print("JSON文件为空或格式不正确（应为一个列表）。")
        return

    # 初始化指标
    correct_predictions = 0
    total_tokens_consumed = 0
    total_samples = len(results_data)

    # 遍历数据进行统计
    for entry in results_data:
        # 检查 "is_correct" 键是否存在且为 True
        if entry.get("is_correct", False):
            correct_predictions += 1
        
        # 累加 "tokens_generated"
        total_tokens_consumed += entry.get("tokens_generated", 0)

    # 计算最终结果
    if total_samples > 0:
        accuracy = (correct_predictions / total_samples) * 100
        avg_token_consumption = total_tokens_consumed / total_samples

        print("--- 结果分析报告 ---")
        print(f"分析文件: {file_path}")
        print(f"总样本数: {total_samples}")
        print(f"正确预测数: {correct_predictions}")
        print(f"准确率 (ACC): {accuracy:.2f}%")
        print(f"总消耗Token数: {total_tokens_consumed}")
        print(f"平均每个样本消耗Token: {avg_token_consumption:.2f}")
    else:
        print("文件中没有找到样本数据。")


if __name__ == "__main__":
    # 设置命令行参数解析器
    parser = argparse.ArgumentParser(
        description="从JSON文件中计算推理结果的准确率和平均Token消耗。"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default="./step5_eval_data/org_inference_results.json",
        help="包含推理结果的JSON文件的路径。"
    )
    
    args = parser.parse_args()
    
    # 执行分析函数
    analyze_inference_results(args.input_file)