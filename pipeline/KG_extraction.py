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
def read_json(file_path):
    """
    读取标准 JSON 文件，返回数据对象
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 解析 JSON 文件并将其转换为 Python 对象（如字典或列表）
    return data
def get_rewrite_prompt(intent,KG):
    relations=KG.get_relations()
    
    sample_kg_str=    "   ".join([f"- <relationship>{relation}" for relation in relations])
    # print(sample_kg_str)
    instruction = f'''
You are a helpful AI assistant, and you will assist me in analyzing the Query. When encountering ambiguous intents or things, you need to use the "KG.search" function to retrieve relevant information from the knowledge graph.

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

You should place your output within <output></output> and enclose the KG.search call in <api_call></api_call>.
Below are some examples for reference:
example1: 
<intent><speak>Speaker: Bob</speak> I'd like to cancel alarms during my dad's preferred dining time. 
</intent>

<output>
<thought>
"my dad's preferred dining time" is an ambiguous concept, but the relationships exist in the knowledge graph. Therefore, we will invoke KG.search call.
</thought>
<api_call>
KG.search(start="Bob", path=["<relationship>father","<relationship>prefer_dinnertime"])
</api_call>
</output>

example2: 
<intent>
<speak>Speaker: Alice</speak> I'd like to book a train ticket from son's preferred city to husband's preferred city.
</intent>

<output>
<thought>
"son's preferred city" and "husband's preferred city" are two ambiguous concepts, but the relationships exist in the knowledge graph. Therefore, we will invoke two KG.search calls.
</thought>
<api_call>
KG.search(start="Alice", path=["<relationship>child","<relationship>prefer_city"])
KG.search(start="Alice", path=["<relationship>husband","<relationship>prefer_city"])
</api_call>
</output>

Explanation of the tags is as follows:

<thought></thought>: Step-by-step thinking about whether ambiguous nouns are included and which tool needs to be called for clarification
<api_call></api_call>: Making the tool call

Please strictly follow the example format for completion generation, and do not generate any additional content.
'''

    prompt = f"<intent>\n{intent}\n</intent>\n"

    return instruction, prompt
def model_rewrite(instruction, prompt, model, tokenizer, model_name):
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
                input_tokens = tokenizer.encode(prompt)
                total_token_length += len(input_tokens)
                prompt_token_length += len(input_tokens)

                # 设置生成参数
                sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.8,
                    repetition_penalty=1.05,
                    max_tokens=4096,
                    stop=["</output>"]
                )

                # 执行推理
                outputs = model.generate([prompt], sampling_params)
                temp_response = outputs[0].outputs[0].text
                output_tokens = tokenizer.encode(temp_response)
                total_token_length += len(output_tokens)
                completion_token_length += len(output_tokens)
                
                # 更新完整响应
                response += temp_response
                
                
                if outputs[0].outputs[0].stop_reason == "</output>":
                    response += f"</output>" 
                    prompt += f"{temp_response}</output>"
                    stop_flag = -1
                else:
                    stop_flag = -1
            else:
                chat_response = model.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=0.7,
                    max_tokens=4096,
                    stop=["</output>"]
                )
                print(chat_response)
                temp_response = chat_response.choices[0].message.content.strip()
                total_token_length += chat_response.usage.total_tokens
                prompt_token_length += chat_response.usage.prompt_tokens
                completion_token_length += chat_response.usage.completion_tokens

                response += temp_response
               
                
                if has_unpaired_tag(response, r'<output>', r'</output>'): # 终止符结束
                    response += f"</output>" 
                    messages[-1]["content"] += f"{temp_response}</output>"
                    stop_flag = -1
                else:
                    stop_flag = -1

    except Exception as e:
        # 获取具体的错误行号和文件名
        stack_trace = ''.join(traceback.format_exception(type(e), e, e.__traceback__))
        rewrite_error = f"Stack Trace:\n{stack_trace}"

    return response, rewrite_error, total_token_length, prompt_token_length, completion_token_length

def rewrite_entry(KG, processed_datas,args):
    modelname=args.model_name.replace("/","_")
    rewrite_result_file = f"results/KG_extraction_results/{time.time()}_{modelname}_intermediate.json"
    rewrite_result = []

        
        
        # rewrite推理
    model, tokenizer = load_model(args.model_name)

            # 分意图进行处理
    for data in tqdm(processed_datas,desc="KG extraction path generating..."):
        # 判断是否需要rewrite
        instruction, prompt = get_rewrite_prompt(data["query"], KG)
        response, error, total_token_length, prompt_token_length, completion_token_length = model_rewrite(instruction, prompt, model, tokenizer, args.model_name)

        data["instruction"] = instruction
        data["prompt"] = prompt
        data["response"] = response
        data["prompt_token_length"] = prompt_token_length
        data["completion_token_length"] = completion_token_length
        data["total_token_length"] = total_token_length
        data["error"] = error
        rewrite_result.append(data)
    # 释放显存
    if tokenizer != None:
        vllm_release_memory(model)
    save_json_file(rewrite_result, rewrite_result_file)
    return rewrite_result
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
        print("未找到括号内的信息")
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
    parser = argparse.ArgumentParser(description="KG extraction")

    str_to_bool = lambda x: x.lower() == 'true'

    parser.add_argument('--model_name', type=str, default="QwQ-32B",  help="extraction model")  
    parser.add_argument('--data_path', type=str, default="datasets/MTU-Bench/data_goldenKG.jsonl",  help="goldjsonl path")
    
    args = parser.parse_args()

    return args
def main():
    args=parse_args()
    
    ###generate KG extraction path
    inputdatas=read_jsonl(args.data_path)
    processed_datas=preprocessdata(inputdatas)
    KG=KnowledgeGraph()
    KG.loadfromdisk("KGs/familykg.txt")
    rewrite_results=rewrite_entry(KG,processed_datas,args)
    pass

if __name__=="__main__":
    main()