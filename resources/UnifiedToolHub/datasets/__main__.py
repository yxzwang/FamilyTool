import argparse
import os
import sys
import tqdm
import requests


from source import *

DATASETS_DIR = os.path.dirname(os.path.abspath(__file__))
DOWNLOADED_DIR = os.path.join(DATASETS_DIR, "downloaded")
PROCESSED_DIR = os.path.join(DATASETS_DIR, "processed")
TOOL_DIR = os.path.join(DATASETS_DIR, "tools")


urls = {
    "API-Bank": "https://huggingface.co/datasets/liminghao1630/API-Bank/resolve/main/",
    "BFCL": "https://raw.githubusercontent.com/ShishirPatil/gorilla/refs/heads/main/berkeley-function-call-leaderboard/data/",
    "ToolACE": "https://huggingface.co/datasets/Team-ACE/ToolACE/resolve/main/",
    "ToolAlpaca": "https://raw.githubusercontent.com/tangqiaoyu/ToolAlpaca/refs/heads/main/data/",
    "TaskBench": "https://raw.githubusercontent.com/microsoft/JARVIS/refs/heads/main/taskbench/",
    "MTU-Bench": "https://raw.githubusercontent.com/MTU-Bench-Team/MTU-Bench/refs/heads/main/MTU-Eval/benchmark/",
    "Seal-Tools": "https://raw.githubusercontent.com/fairyshine/Seal-Tools/refs/heads/master/Seal-Tools_Dataset/",
    "ToolHop": "https://huggingface.co/datasets/bytedance-research/ToolHop/resolve/main/data/",
}

download_files = {
    "BFCL": [
        # "BFCL_v3_multi_turn_base.json",
        # "BFCL_v3_multi_turn_composite_unused.json",
        # "BFCL_v3_multi_turn_long_context.json",
        # "BFCL_v3_multi_turn_miss_func.json",
        # "BFCL_v3_multi_turn_miss_param.json",
        # "BFCL_v3_java.json",
        # "BFCL_v3_javascript.json",
        "BFCL_v3_multiple.json",
        "BFCL_v3_parallel.json",
        "BFCL_v3_parallel_multiple.json",
        "BFCL_v3_simple.json",
        # "BFCL_v3_sql_unused.json",
        "BFCL_v3_live_multiple.json",
        "BFCL_v3_live_parallel.json",
        "BFCL_v3_live_parallel_multiple.json",
        "BFCL_v3_live_simple.json",
        # "BFCL_v3_chatable.json",
        # "BFCL_v3_exec_multiple.json",
        # "BFCL_v3_exec_parallel.json",
        # "BFCL_v3_exec_parallel_multiple.json",
        # "BFCL_v3_exec_simple.json",
        # "BFCL_v3_irrelevance.json"
        # "BFCL_v3_live_irrelevance.json",
        # "BFCL_v3_live_relevance.json",
        # "BFCL_v3_rest.json",
    ],
    "ToolACE": [
        "data.json"
    ],
    "ToolAlpaca": [
        "train_data.json"
    ],
    "TaskBench": {
        "data_dailylifeapis": [
            "data.json",
            "graph_desc.json",
        ],
        # # 目前看 huggingface 和 multimedia 的数据集格式和通用工具使用有一定差别
        # "data_huggingface":[
        #     "data.json",
        #     "graph_desc.json",
        # ],
        # "data_multimedia":[
        #     "data.json",
        #     "graph_desc.json",
        # ],
    },
    "MTU-Bench": [
        "M-M_eval.jsonl",
        "M-S_eval.jsonl",
        "S-M_eval.jsonl",
        "S-S_eval.jsonl",
        "OOD_eval.jsonl",
    ],
    "Seal-Tools": [
        "tool.jsonl",
        "dev.jsonl",
        "train.jsonl",
        "test_in_domain.jsonl",
        "test_out_domain.jsonl",
    ],
    "ToolHop": [
        "ToolHop.json"
    ],
    "API-Bank": {
        "training-data": [
            "lv1-api-train.json",
            "lv1-response-train.json",
            "lv1-train.json",
            "lv2-api-train.json",
            "lv2-response-train.json",
            "lv2-train.json",
            "lv3-api-train.json",
            "lv3-response-train.json",
            "lv3-train.json"
        ],
        "test-data": [
            "level-1-api.json",
            "level-1-response.json",
            "level-2-api.json",
            "level-2-response.json",
            "level-3-batch-inf-icl.json",
            "level-3-batch-inf-response.json",
            "level-3-batch-inf.json",
            "level-3.json",
        ]
    },
}

process_method = {
    "TaskBench": process_daily_life_apis,
    "ToolAlpaca": process_tool_alpaca,
    "MTU-Bench": process_mtu_bench,
    "BFCL": process_some,
    "ToolACE": process_tool_ace,
    "Seal-Tools": process_seal_tools,
    "API-Bank": process_data_from_original_dataset,
}


def download_file(url, file_path):
    # 如果文件已存在，提示用户是否覆盖
    if os.path.exists(file_path):
        print(f"文件 {file_path[len(DOWNLOADED_DIR):]} 已存在，跳过。")
        return
    else:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # 发起 GET 请求，打开流式下载模式
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()  # 若响应出错，会抛出异常

            # 根据响应头获取文件大小（字节数）
            total_size = int(response.headers.get("content-length", 0))
            block_size = 8192  # 每次读取的块大小，可根据需要调整

            # 使用 tqdm 包裹文件下载，显示进度条
            with open(file_path, "wb") as file:
                # 使用 tqdm 显示进度条，total_size 是文件总大小，block_size 是每次读取的块大小
                with tqdm.tqdm(total=total_size, unit='B', unit_scale=True,
                            desc="开始下载 " + file_path[len(DOWNLOADED_DIR):]) as pbar:
                    for data in response.iter_content(chunk_size=block_size):
                        file.write(data)
                        pbar.update(len(data))  # 更新进度条  
        print("下载完成！")
    except Exception as e:
        print("下载失败！", str(e))



def download_one_dataset(dataset_name):
    if urls[dataset_name] == "Too Large.":
        print(f"数据集 {dataset_name} 太大，请按照 datasets/downloaded/{dataset_name}/README.md 中的步骤自行下载。")
        return False
    folder_path = os.path.join(DOWNLOADED_DIR, dataset_name)
    os.makedirs(folder_path, exist_ok=True)
    if isinstance(download_files[dataset_name], list):
        for name in download_files[dataset_name]:
            download_file(urls[dataset_name] + name, os.path.join(folder_path, name))
    else:
        for path, name_list in download_files[dataset_name].items():
            os.makedirs(os.path.join(folder_path, path), exist_ok=True)
            for name in name_list:
                download_file(urls[dataset_name] + path + "/" + name, os.path.join(folder_path, path, name))
    if dataset_name == "BFCL":
        os.makedirs(os.path.join(folder_path, "possible_answer"), exist_ok=True)
        for name in download_files[dataset_name]:
            download_file(urls[dataset_name] + "possible_answer/" + name,
                          os.path.join(folder_path, "possible_answer", name))
    return True



def setup_argparse():
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(description='数据集下载和处理工具')
    
    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 下载数据集命令
    download_parser = subparsers.add_parser('download', help='下载指定数据集')
    download_parser.add_argument('datasets', nargs='+', choices=urls.keys(), 
                               help='要下载的数据集名称，可指定多个')
    
    # 处理数据集命令
    process_parser = subparsers.add_parser('process', help='处理指定数据集')
    process_parser.add_argument('datasets', nargs='+', choices=process_method.keys(), 
                              help='要处理的数据集名称，可指定多个')
    
    # 下载并处理数据集命令
    deal_parser = subparsers.add_parser('deal', help='下载并处理指定数据集')
    deal_parser.add_argument('datasets', nargs='+', choices=process_method.keys(), 
                           help='要下载并处理的数据集名称，可指定多个')
    
    return parser

def main():
    parser = setup_argparse()
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    for dataset in args.datasets:
        
        if args.command == 'download':
            print(f"\n开始下载数据集: {dataset}")
            download_one_dataset(dataset)
        
        elif args.command == 'process':
            print(f"\n开始处理数据集: {dataset}")
            os.makedirs(os.path.join(PROCESSED_DIR, dataset), exist_ok=True)
            os.makedirs(os.path.join(TOOL_DIR, dataset), exist_ok=True)
            process_method[dataset](
                os.path.join(DOWNLOADED_DIR, dataset), 
                os.path.join(PROCESSED_DIR, dataset), 
                os.path.join(TOOL_DIR, dataset)
            )
        
        elif args.command == 'deal':
            print(f"\n开始下载并处理数据集: {dataset}")
            if download_one_dataset(dataset):
                os.makedirs(os.path.join(PROCESSED_DIR, dataset), exist_ok=True)
                os.makedirs(os.path.join(TOOL_DIR, dataset), exist_ok=True)
                process_method[dataset](
                    os.path.join(DOWNLOADED_DIR, dataset), 
                    os.path.join(PROCESSED_DIR, dataset), 
                    os.path.join(TOOL_DIR, dataset)
                )

if __name__ == "__main__":
    main()