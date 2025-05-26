import argparse
import json
import os
import random
import traceback
import numpy as np
from tqdm import tqdm
from vllm import SamplingParams
from utils import *
import time
from datetime import datetime
from KG_extraction_post_process_make_intermediate_json import KG_extraction_post
def get_rewrite_prompt(intent,KG):
    relations=KG.get_relations()
    
    sample_kg_str=    "   ".join([f"- <relationship>{relation}" for relation in relations])
    # print(sample_kg_str)
    instruction = f'''
You are a helpful AI assistant, and you will assist me in analyzing the Query. When encountering ambiguous intents or things, you need to use the "KG.search" python function to retrieve relevant information from the knowledge graph.

def KG.search(start: str, path: List[str]) -> List[Tuple[str, str, str]]:
    """Search the knowledge graph for user relationships, locations, preferences, and other relevant information.  

    This API supports multi-hop queries to discover relationship links between users, as well as associations between users and locations or preferences.  

    **Args:**  
    - **start**: The starting point of the knowledge graph query. If related to personal relationships, the speaker in `<speak></speak>` tags can be referenced.  
    - **path**: The reasoning path in the knowledge graph, where each element must be a predefined relation type. Only use relations from **Valid Relations**.  

    **Returns:**  
    - **List[Tuple[str, str, str]]**: A list of triples containing the complete relationship paths.
        
    Valid Relations :
    {sample_kg_str}
       
        
    Notes:

    All relations must come from the predefined list, including the content within <> (which should also be generated).

    Each hop in the query path must be a valid relation.

    The returned results will include the complete relationship path.

    Fabricating non-existent relations is prohibited.
    """
    pass


Below are some examples for reference:
example 1: 
User query: <speak>Speaker: Bob</speak> I'd like to cancel alarms during my dad's preferred dining time. 

Output:

```python 
KG.search(start="Bob", path=["<relationship>father","<relationship>prefer_dinnertime"])
```

example 2: 
User query: <speak>Speaker: Alice</speak> I'd like to book a train ticket from son's preferred city to husband's preferred city.

Output:

```python
KG.search(start="Alice", path=["<relationship>child","<relationship>prefer_city"])
KG.search(start="Alice", path=["<relationship>husband","<relationship>prefer_city"])
```

Please strictly follow the example format for completion generation, and do not generate any additional content.
'''

    prompt = intent

    return instruction, prompt
def model_rewrite(instruction, prompt, model, tokenizer, args):
    model_name=args.model_name
    # 准备批量输入
    messages = [
        {
            "role": "system",
            "content": instruction
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    if args.fast_run:
        return messages

    # 初始化生成结果
    response = ""
    rewrite_error = ""
    stop_flag = 1
    total_token_length = 0
    prompt_token_length = 0
    completion_token_length = 0
    # 尝试使用文本补全的的方式来进行推理
    prompt = f"{instruction}\n{prompt}"
    try:
        while stop_flag > 0:
            if tokenizer != None:
                # 编码当前输入
                if args.use_template:
                    messages = [{"role": "user", "content": prompt}]
                    if "Qwen3" in model_name:
                        if args.no_think:
                            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,enable_thinking=False) ### enable_thinking=False to turn off thinking
                        else:
                            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    else:
                        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                else:
                    if "Qwen3" in model_name:
                        if args.no_think:
                            prompt=prompt+"<think>\n\n</think>\n\n"###no think for qwen3#
                input_tokens = tokenizer.encode(prompt)
                total_token_length += len(input_tokens)
                prompt_token_length += len(input_tokens)

                # 设置生成参数
                sampling_params = SamplingParams(
                    temperature=0.7,
                    # top_p=0.8,
                    repetition_penalty=1.05,
                    max_tokens=4096,
                )

                # 执行推理
                outputs = model.generate([prompt], sampling_params)
                temp_response = outputs[0].outputs[0].text
                output_tokens = tokenizer.encode(temp_response)
                total_token_length += len(output_tokens)
                completion_token_length += len(output_tokens)
                
                # 更新完整响应
                response += temp_response
                
                

                stop_flag = -1
            else:
                chat_response = model.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4096,
                )
                print(chat_response)
                temp_response = chat_response.choices[0].message.content.strip()
                total_token_length += chat_response.usage.total_tokens
                prompt_token_length += chat_response.usage.prompt_tokens
                completion_token_length += chat_response.usage.completion_tokens

                response += temp_response
               
                

                stop_flag = -1

    except Exception as e:
        # 获取具体的错误行号和文件名
        stack_trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        rewrite_error = f"Stack Trace:\n{stack_trace}"

    return response, rewrite_error, total_token_length, prompt_token_length, completion_token_length

def rewrite_entry(KG, processed_datas,args):
    modelname=args.model_name.replace("/","_")
    if "Qwen3" in modelname:
        if args.no_think:
            no_think="nothink"
        else:
            no_think=""
    else:
        no_think=""
    rewrite_result_file = f"results/KG_extraction_results/{datetime.now()}_{modelname}_{args.data_path.split("/")[-1][:-6]}_usetemplate={args.use_template}_{no_think}_intermediate.json"
    rewrite_result = []

        
        
        # rewrite推理
    model, tokenizer = load_model(args.model_name)

    sampling_params = SamplingParams(
        temperature=0.7,
        # top_p=0.8,
        repetition_penalty=1.05,
        max_tokens=4096,
    )

    if args.fast_run:
        print("fast run")
        chat_list = []
        instruction_list, prompt_list = [], []
        for data in tqdm(processed_datas,desc="KG extraction path generating..."):
            instruction, prompt = get_rewrite_prompt(data["query"], KG)
            chat_list.append(model_rewrite(instruction, prompt, model, tokenizer, args))
            instruction_list.append(instruction)
            prompt_list.append(prompt)
        output_list = model.chat(chat_list, sampling_params)
        for i, output in enumerate(output_list):
            response = output.outputs[0].text
            output_tokens = tokenizer.encode(response)
            input_tokens = tokenizer.encode(prompt_list[i])
            prompt_token_length = len(input_tokens)
            total_token_length = len(output_tokens)
            completion_token_length = len(output_tokens)
            data = {
                "instruction": instruction_list[i],
                "prompt": prompt_list[i],
                "response": response,
                "prompt_token_length": prompt_token_length,
                "completion_token_length": completion_token_length,
                "total_token_length": total_token_length,
                "error": ""
            }
            rewrite_result.append(data)
    else:
            # 分意图进行处理
        for data in tqdm(processed_datas,desc="KG extraction path generating..."):
            # 判断是否需要rewrite
            instruction, prompt = get_rewrite_prompt(data["query"], KG)
            response, error, total_token_length, prompt_token_length, completion_token_length = model_rewrite(instruction, prompt, model, tokenizer, args)

            data["instruction"] = instruction
            data["prompt"] = prompt
            data["response"] = response
            data["prompt_token_length"] = prompt_token_length
            data["completion_token_length"] = completion_token_length
            data["total_token_length"] = total_token_length
            data["error"] = error
            rewrite_result.append(data)
    # 释放显存
    # if tokenizer != None:
    #     vllm_release_memory(model)
    save_json_file(rewrite_result, rewrite_result_file)
    return rewrite_result,rewrite_result_file
def transform_query_with_kg(original_query):
    """
    Extract kg information from parentheses and create a new query.
    
    Args:
    original_query (str): The original query string
    
    Returns:
    tuple: (new_query, kg_info)
    """
    # Find text within parentheses
    kg_match = re.search(r'\((.*?)\)', original_query)
    
    if kg_match:
        kg_info = kg_match.group(1)
        # Create new query by appending kg information
        sub_query=original_query.replace('('+kg_info+')',"")
        # new_query = f"{sub_query} The extra information for the query is ({kg_info})."
        return sub_query, kg_info
    else:
        print("no found")
        print(original_query)
        return original_query, ""

def preprocessdata(inputdatas):
    newdata=[]
    for data in inputdatas:
        inputdict=data[2] ##### "role": "user"的dict
        goldquery=inputdict["content"]
        inputdict["query"]=goldquery.split("The extra information for the query")[0].strip()
        assert len(inputdict["query"])>10
        newdata.append(inputdict)
    return newdata
def read_jsonl(file_path):
    """
    读取 JSONL 文件，每行是一个 JSON 对象，返回一个列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))  # 将每一行的 JSON 数据解析并添加到列表中
    return data

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="")


    parser.add_argument('--model_name', type=str, default="Qwen2.5-7B-Instruct",  help="model name")  
    parser.add_argument('--data_path', type=str, default="datasets/familytool-b.jsonl",  help="goldjsonl path")
    parser.add_argument('--kg_path', type=str, default="KGs/familykg-b.txt",  help="kg path")
    parser.add_argument('--k', type=int, default=3, help="top k")
    parser.add_argument('--KGretrieval_type', type=str, default="exact", help="",choices=["exact","relation_retrieval"])
    parser.add_argument('--use_template', type=bool,default=True, help="whether to use template")  
    parser.add_argument('--no_think', action="store_true",default=False, help="for qwen3")  
    parser.add_argument('--fast_run', action="store_true",default=False, help="for qwen3")  
    parser.add_argument('--useprompt', type=str, default="new_prompt", help="new prompt or old ones")
    args = parser.parse_args()

    return args


def main():
    args=parse_args()
    
    ###generate KG extraction path
    if "no_think" in args.model_name:
        args.no_think=True
    inputdatas=read_jsonl(args.data_path)
    processed_datas=preprocessdata(inputdatas)
    KG=KnowledgeGraph()
    KG.loadfromdisk(args.kg_path)
    rewrite_results,extraction_file_path=rewrite_entry(KG,processed_datas,args)
    KG_extraction_post(extraction_file_path,args.KGretrieval_type,args.data_path,args.kg_path,args)

if __name__=="__main__":
    main()