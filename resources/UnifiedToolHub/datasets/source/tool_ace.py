import os
import re
import json


def format_data(raw_data):
    
    new_k = []  # new_k：用于存储一条完整的、转换完格式的数据

    for v in raw_data:
        new_v = {}  # new_v：用于保存一个Agent的内容，包含 "role" 和 "content" 两个字段
        
        # User Agent：用户提问
        if v['from'] == 'user':
            new_v['role'] = "user"
            new_v['content'] = v['value']
            new_k.append(new_v)
        
        # Assistant Agent：（1）回答用户问题； 或（2）提供需要调用的 API 名称和参数
        elif v['from'] == 'assistant':
            # 如果 Assistant Agent 的 value 值是以列表的形式存储的，表示（2）
            if v['value'].startswith('['):
                matches = re.findall(r"([\w\s]+)\((.*?)\)", v["value"])
                if matches:
                    tools = []
                    for match in matches:
                        api_name = match[0]
                        params_string = match[1]
                        # 将参数字符串解析为字典
                        params = dict(re.findall(r'(\w+)=["\'](.*?)["\']', params_string))
                        tools.append({
                        "name": api_name,
                        "parameters": params
                    })
                    # 将解析后的工具调用附加到结果中
                    new_v['role'] = "tool_call"
                    new_v['content'] = tools 
            else:
                # 如果 Assistant Agent 的 value 值不是是以列表的形式存储的，而是字符串的形式存储的，则表示（1）
                new_v['role'] = "assistant"
                new_v['content'] = v['value']
            new_k.append(new_v)
        
        # Tool Agent
        elif v['from'] == 'tool':
            api_call_counts = {}    # 用于记录某个 API 在当前轮次调用的次数
            new_v['role'] = 'tool_response'
            new_v['content'] = {}
            for calls in json.loads(v['value']):
                api_name = calls['name']
                results = calls['results']
                if api_name not in api_call_counts:
                    api_call_counts[api_name] = 0
                key = f"{api_name}.{api_call_counts[api_name]}"     # 为了区分在一轮调用中，多次调用同一个API的情况
                new_v['content'][key] = results
                api_call_counts[api_name] += 1
            new_k.append(new_v)
    return new_k

def process_tool_ace(from_path, to_path, tool_path):
    data_path = os.path.join(from_path, "data.json")
    save_path = os.path.join(to_path, "tool_ace_processed.jsonl")

    with open(data_path, 'r', encoding='utf-8') as file:
        all_data = json.load(file)

    processed_data = []      # 用于保存格式化后的数据
    for index, data in enumerate(all_data):
        conversations = data['conversations']
        processed_data.append(format_data(conversations))


    tool_dict = {}
    for data in processed_data:
        for i, item in enumerate(data):
            if len(item) == 0: 
                # 有问题的数据
                break
            if item["role"] == "tool_call" and data[i-1]["role"] == "user":
                for call in item["content"]:
                    if call["name"] not in tool_dict:
                        tool_dict[call["name"]] = []
                    tool_dict[call["name"]].append({
                        "query": data[i-1]["content"],
                        "tool_call": item["content"]
                    })
        if len(data) == 2 and len(data[-1]) == 0:
            data.pop()

    processed_data = [data for data in processed_data if len(data)>1]

    tool_list = []
    cnt = 0
    for k, v in tool_dict.items():
        cnt += len(v)
        tool_list.append({
            "name": k,
            "demo": v,
        })

    print("工具数量",len(tool_dict), "平均每工具样本量", round(cnt/len(tool_dict),2))
    data_list = [json.dumps([{
        "role": "id",
        "content": f"ToolACE_{index}"
    }]+data, ensure_ascii=False) for index, data in enumerate(processed_data)]
    print("样本总数", len(data_list), "（一个样本中可能使用多个工具）")

    with open(save_path, 'w', encoding='utf-8') as file:
        file.write("\n".join(data_list))
        print(save_path, "saved.")

    tool_path = os.path.abspath(os.path.join(tool_path, "tools_with_demos.jsonl"))
    with open(tool_path, 'w', encoding='utf-8') as file:
        file.write("\n".join([json.dumps(tool, ensure_ascii=False) for tool in tool_list]))
        print(tool_path, "saved.")