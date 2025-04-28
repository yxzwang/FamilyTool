test_models = [
    #     dict(
    #     path="deepseek-chat",
    #     api_key="", base_url="",

    # ),
    #  dict(
    #     path="deepseek-reasoner",
    #     api_key="", base_url="",
    #     max_workers=4,
    #     sampling_params=dict(
    #         temperature = 0.7,
    #         max_tokens=8192,
    #     ),
    # ),
    #     dict(
    #     path="o3-mini",
    #     api_key="", base_url="",
    #     sampling_params=dict(
    #         temperature = 0.7,
    #         max_tokens=4096,
    #     ),
    #     tool_choice="required",
    # ),
    # dict(
    #     path="gpt-4o",
    #     api_key="", base_url="",
    #     sampling_params=dict(
    #         temperature = 0.7,
    #         max_tokens=4096,
    #     ),
    #     tool_choice="required", # default: auto
    #     max_workers=4, # default: 1
    # ),
    # dict(
    #     type="Qwen_2_5",
    #     path="QwQ-32B",
    #     tp=2,
    #     sampling_params=dict(
    #         max_tokens=4096,
    #     )
    # ),
    dict(
        path="Qwen2.5-7B-Instruct",
       
        tp=1,
        # max_model_length=8000,
        sampling_params=dict(
            max_tokens=4096,
        )
    ),
    # dict(
    #     type="Llama_3_1",
    #     path="Meta-Llama-3.1-8B-Instruct",
    #     sampling_params=dict(
    #         max_tokens=1024,
    #     ),
    #     # max_model_len=4096,
    #     tp=1,
    # ),
]
test_datasets = [
    # "API-Bank",
    # "BFCL",
    # "MTU-Bench",
    # "Seal-Tools",
    # "TaskBench",
    # "ToolACE",
    # "ToolAlpaca",
    "../../datasets/MTU-Bench/data_goldenKG.jsonl",


]
test_mode = "single_last"
# - multiple 单轮工具使用测试
# - single_* 把所有数据当成单轮工具使用
#   - single_first 以第一个 tool_call 块为答案
#   - single_last 以最后个 tool_call 块为答案
#   - single_all 以 teacher forcing 方式的测试每个 tool_call 块
test_with_tag = [
    # "single_round",
]
test_metrics = [
    "ExactMatch",
    "ToolAccuracy",
    "ParameterAccuracy",
]
test_strategy = "sequential" # "sequential" or "parallel" # 智能显卡分配，过于高端暂未开发
save_strategy = dict(
    save_log=False, # 测试过程中记录 log # 还没开发
    save_output=False, # 记录模型原始的输出
    save_input=False, # 记录模型原始的输入
    save_result=True, # 记录按照 think, content, tool_calls 分隔后的结果
    save_golden_answer=True, # 记录 golden_answer
    save_path="./results",
    with_timestamp=True,
    only_date=True,
)
report_strategy = [
    "json",
]
json_confg = dict(
    path="./results",
)
