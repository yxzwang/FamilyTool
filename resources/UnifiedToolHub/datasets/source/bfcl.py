import os
import json
from collections import defaultdict


key_list = ["simple", "multiple", "parallel", "parallel_multiple"]
code = ["java", "javascript"]
multi_turn = ["base", "composite", "long_context", "miss_func", "miss_param"]

def read_one_dataset(tag, key, from_path):
    tools_set = set()
    all_data = []
    path = os.path.join(from_path, "BFCL_{}_{}.json".format(tag, key))
    ans_path = os.path.join(from_path, "possible_answer", "BFCL_{}_{}.json".format(tag, key))
    question_lines = open(path).readlines()
    ans_lines = open(ans_path).readlines()
    for i, (question, ans) in enumerate(zip(question_lines, ans_lines)):
        question = json.loads(question)
        ans = json.loads(ans)
        query = question["question"]
        tools = question["function"]
        for tool in tools:
            tools_set.add(json.dumps(tool, ensure_ascii=False))
        bfcl_id = question["id"]
        if ans["id"] == "live_multiple_1052-279-0" and bfcl_id == "live_multiple_1052-79-0":
            # 数据有 bug
            bfcl_id = ans["id"]
        else:
            assert ans["id"] == bfcl_id, ans["id"] + bfcl_id
        ans = [
            {
                "name": list(call.items())[0][0],
                "parameters": list(call.items())[0][1],
                "depend_on": []
            } for call in ans["ground_truth"]
        ]
        assert len(query) == 1
        if isinstance(query[0], list):
            query = query[0]
        if len(query) == 2:
            system = query[0]["content"]
            query = query[1]["content"]
        else:
            system = ""
            query = query[0]["content"]
        all_data.append([
            {
                "role": "id",
                "content": "BFCL_v3_"+bfcl_id,
            }, {
                "role": "system",
                "content": system,
            }, {
                "role": "candidate_tools",
                "content": tools,
            }, {
                "role": "user",
                "content": query,
            }, {
                "role": "tool_call_ground_truth",
                "content": ans,
            }
        ])
    return all_data, tools_set


def read_multi_turn_dataset(tag, key, from_path):
    category_stats = defaultdict(int)
    tool_stats = defaultdict(int)
    all_data = []
    path = os.path.join(from_path, "BFCL_{}_{}.json".format(tag, key))
    ans_path = os.path.join(from_path, "possible_answer", "BFCL_{}_{}.json".format(tag, key))
    question_lines = open(path).readlines()
    ans_lines = open(ans_path).readlines()
    for i, (question, ans) in enumerate(zip(question_lines, ans_lines)):
        question = json.loads(question)
        ans = json.loads(ans)
        query = question["question"]
        bfcl_id = question["id"]
        ans = ans["ground_truth"]
        if "involved_classes" in question and "path" in question:
            categories = question["involved_classes"]
            tools = question["path"]
            # 统计类别
            for category in categories:
                category_stats[category] += 1
            # 统计工具
            for tool in tools:
                tool_stats[tool] += 1
    return category_stats, tool_stats



def calculate_statistics(tag, key):
    all_data = read_one_dataset(tag, key)
    
    # 计算统计信息
    num_samples = len(all_data)
    avg_tools = sum(len(data["tools"]) for data in all_data) / num_samples if num_samples > 0 else 0
    avg_answers = sum(len(data["answer"]) for data in all_data) / num_samples if num_samples > 0 else 0
    
    print(f"\n统计信息 - {tag}_{key}:")
    print(f"样本数量: {num_samples}")
    print(f"平均工具数: {avg_tools:.2f}")
    print(f"平均答案数: {avg_answers:.2f}")
    
    return {
        "num_samples": num_samples,
        "avg_tools": avg_tools,
        "avg_answers": avg_answers
    }

def get_dict_depth(d):
    if not isinstance(d, (dict, list)):
        return 0
    
    if isinstance(d, list):
        if not d:  # 空列表
            return 0
        return max(get_dict_depth(item) for item in d)
        
    if isinstance(d, dict):
        if not d:  # 空字典
            return 1
        return 1 + max(get_dict_depth(v) for v in d.values())


def flatten_dict(d, parent_key='', sep='.'):
    """将嵌套字典转换为扁平格式"""
    items = []
    if isinstance(d, list):
        return d
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            if any(isinstance(item, dict) for item in v):
                for i, item in enumerate(v):
                    if isinstance(item, dict):
                        items.extend(flatten_dict(item, f"{new_key}[{i}]", sep=sep).items())
                    else:
                        items.append((f"{new_key}[{i}]", item))
            else:
                items.append((new_key, v))
        else:
            items.append((new_key, v))
    return dict(items)

def process_some(from_path, to_path, tool_path):
    stats = {}
    tools_set = set()
    # normal
    for tag in ["v3", "v3_live"]:
        stats[tag] = {}
        for key in key_list:
            all_data, tools = read_one_dataset(tag, key, from_path)
            tools_set.update(tools)
            filename = f"{to_path}/BFCL_{tag}_{key}.jsonl"
            with open(filename, 'w', encoding='utf-8') as file:
                file.write("\n".join(
                    [json.dumps(data, ensure_ascii=False) for data in all_data]
                ))
                print(filename, "saved.")
    # code
    tag = "v3"
    for key in code:
        all_data, tools = read_one_dataset(tag, key, from_path)
        tools_set.update(tools)
        filename = f"{to_path}/BFCL_{tag}_{key}.jsonl"
        with open(filename, 'w', encoding='utf-8') as file:
            file.write("\n".join(
                [json.dumps(data, ensure_ascii=False) for data in all_data]
            ))
            print(filename, "saved.")

    # multi_turn 目前仅完成了统计部分
    # total_category_stats = defaultdict(int)
    # total_tool_stats = defaultdict(int)
    # tag = "v3_multi_turn"
    # for key in multi_turn:
    #     category_stats, tool_stats = read_multi_turn_dataset(tag, key)
    #     for category, count in category_stats.items():
    #         total_category_stats[category] += count 
    #     for tool, count in tool_stats.items():
    #         total_tool_stats[tool] += count

    # print("\n=== 多轮对话统计 ===")
    # print(f"类别总数: {len(total_category_stats)}")
    # print(f"工具总数: {len(total_tool_stats)}")
    # print("\n类别分布:")
    # for category, count in sorted(total_category_stats.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{category}: {count}")
    # print("\n工具分布:")
    # for tool, count in sorted(total_tool_stats.items(), key=lambda x: x[1], reverse=True):
    #     print(f"{tool}: {count}")

    with open(os.path.join(tool_path, "tools_with_doc.jsonl"), 'w', encoding='utf-8') as file_out:
        file_out.write("\n".join([tool for tool in tools_set]))
        print(os.path.join(tool_path, "tools_with_doc.jsonl"), "saved.")