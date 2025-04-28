from datetime import date
from vllm.entrypoints.openai.tool_parsers import Llama3JsonToolParser

from .base import BaseFormatter

class Llama_3_1(BaseFormatter):

    SYSTEM_PROMPT = (
        f"Cutting Knowledge Date: December 2023\n"
        f"Today Date: {date.today().strftime('%d %b %Y')}\n\n"
        "When you receive a tool call response, use the output to format an answer to the original user question.\n"
        "You are a helpful assistant with tool calling capabilities."
    )

    SAMPLING_PARAMS = {
        "temperature": 0.5,
        "top_p": 0.9,
        "max_tokens": 512,
        "repetition_penalty": 1.05,
        "stop": ["<eot_id>"]
    }

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.praser = Llama3JsonToolParser(tokenizer)
    
    def get_prompt(self, messages, candidate_tools):
        new_messages = []

        for message in messages:
            if message["role"] == "tool_call":
                new_messages.append({
                    "role": "assistant",
                    "content": str(message["content"][0] if len(message["content"])==1 else message["content"])
                })
            elif message["role"] == "tool_response":
                for value in message["content"].values():
                    new_messages.append({
                        "role": "tool",
                        "content": value
                    })
            else:
                new_messages.append(message)
        
        prompt = self.tokenizer.apply_chat_template(
            new_messages, 
            tools=candidate_tools, 
            tokenize=False,
            add_generation_prompt=True,
        )

        from_text = """Environment: ipython
Cutting Knowledge Date: December 2023
Today Date: 26 Jul 2024"""

        to_text = f"""Cutting Knowledge Date: December 2023
Today Date: {date.today().strftime('%d %b %Y')}"""

        prompt = prompt.replace(from_text, to_text)

        return prompt
    
    def get_tool_call(self, output):
        if "<|python_tag|>" in output:
            content, tool_calls = output.split("<|python_tag|>")
            if "</think>" in self.tokenizer.get_vocab():
                if "</think>" in content:
                    think, content = content.split("</think>")
            else:
                think = ""
            tool_calls = self.safe_parse_arguments(tool_calls, default_value=[])
            if not isinstance(tool_calls, list):
                tool_calls = [tool_calls]
            return {
                "think": think.strip(),
                "content": content.strip(),
                "tool_calls": tool_calls
            }
        else:
            return super().get_tool_call(output)
        