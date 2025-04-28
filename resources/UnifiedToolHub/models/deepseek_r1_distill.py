from datetime import date

import json
import re
from .base import BaseFormatter

def extract_json_objects(text):
    objects = []
    pos = 0
    index_start=-1
    index_end=-1
    while pos < len(text):
        if text[pos] == '{':
            try:
                obj, idx = json.JSONDecoder().raw_decode(text[pos:])
                if isinstance(obj, dict) and 'name' in obj:
                    objects.append(obj)
                    if index_start==-1:
                        index_start=pos
                    index_end=pos+idx
                pos += idx
            except json.JSONDecodeError:
                pos += 1
        else:
            pos += 1
    return objects, index_start, index_end

def clean_markdown_json_blocks(text):
    pattern = r'```json\n|```'
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

class DeepSeek_R1_Distill(BaseFormatter):

    SYSTEM_PROMPT =r"""
Above are some tools, use them to address query for user. Generate only function calls in json format without other texts after </think>.
For each function call, return a json object with function name and arguments:
{"name": <function-name>, "parameters": <args-json-object>}
"""

    SAMPLING_PARAMS = {
        "temperature": 0.6,
        "top_p": 0.95,
        "max_tokens": 2000,
        "repetition_penalty": 1.05,
        "stop": ["<｜end▁of▁sentence｜>"]
    }

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def get_prompt(self, messages, candidate_tools):
        new_messages = []
        sys_with_tools=f"Current Date: {date.today().strftime('%Y-%m-%d')}"+str(candidate_tools)+self.SYSTEM_PROMPT
        new_messages.append({"role":"system","content":sys_with_tools})
        
        for message in messages:
            if message["role"] == "tool_call":
                
                new_messages.append({
                    "role": "assistant",
                    "content": "\n".join(
                        f"<tool_call>{call}</tool_call>".replace("parameters", "arguments")
                        for call in message["content"]
                    )
                })
            elif message["role"] == "tool_response":
                for k, v in message["content"].items():
                    new_messages.append({
                        "role": "tool",
                        "content": str(v)
                    })
            else:
                new_messages.append(message)
        

        prompt = self.tokenizer.apply_chat_template(
            new_messages,  
            tokenize=False,
            add_generation_prompt=True,
        )

        return prompt
    
    def get_tool_call(self, output):
        if "</think>" in output:
            try:
                think,answer=output.split('</think>')
            except:
                print("not only one </think>")
                think=output.split('</think>')[0]
                answer=output.split('</think>')[-1]
        else:
            think=""
            answer=output

        tool_calls, start, end=extract_json_objects(answer)

        if tool_calls==[]:
            content=answer
        else:
            content=answer[0:start]+answer[end+1:]
        content=clean_markdown_json_blocks(content)

        for call in tool_calls:
            try:
                if "arguments" in call:
                    call["parameters"] = call.pop("arguments")
            except:
                pass
        return {
            "think": think.strip(),
            "content": content.strip(),
            "tool_call": tool_calls
        }