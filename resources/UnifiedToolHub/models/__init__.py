from .llama_3_1 import Llama_3_1
from .qwen_2_5 import Qwen_2_5
from .deepseek_r1 import DeepSeek_R1
from .deepseek_r1_distill import DeepSeek_R1_Distill
from .api_requester import APIRequester

__all__ = [
    "Llama_3_1",
    "Qwen_2_5",
    "APIRequester",
    "DeepSeek_R1",
    "DeepSeek_R1_Distill",
]

lowercase_mapping = {
    "Llama_3_1": [
        "llama3.1",
        "llama-3.1",
        "llama_3_1",
        "llama_31",
        "llama31",
        "llama3_1"
    ],
    "Qwen_2_5": [
        "qwq",
        "qwen2.5",
        "qwen-2.5",
        "qwen_2_5",
        "qwen_25",
        "qwen25",
        "qwen2_5"
    ],
    "APIRequester": [
        "gpt-4o",
        "gpt_4o",
        "o3-mini",
        "deepseek-chat",
        "deepseek-reasoner"
    ],
    "DeepSeek_R1": [
        "deepseek-r1",
        "deepseek_r1",
        "deepseekr1",   
    ],
    "DeepSeek_R1_Distill": [
        "deepseek-r1-distill",
        "deepseek_r1_distill",
        "deepseekr1_distill",   
        "deepseekr1-distill",
    ],
}
