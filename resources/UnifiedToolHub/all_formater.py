from datetime import date
import json


class ToolChatFormat:

    MY_SYSTEM_PROMPT = (
        "You are an expert in invoking function to solve the problem. "
        "You are given a list of functions in JSON format and a question. "
        "Based on the question, you need to make a helpful function call. "
        'You should ONLY give the function call in JSON format like {"name": "API", "parameters":{} }'
    )

    @staticmethod
    def dialog_template(query, tools_with_doc):
        """
        这是我们编写的 llama2 和 llama3 使用工具的模板（因为官方没有提供模板）
        """
        return [
            {
                "role": "system",
                "content": ToolChatFormat.MY_SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": (
                    f"Here is a list of functions in JSON format: {json.dumps(tools_with_doc)}\n"
                    f"Here is the question: {query}\n"
                    "You should give a JSON-format function call WITHOUT any extra text."
                )
            }
        ]

    @staticmethod
    def post_process(output):
        return output


class GroqChatFormat(ToolChatFormat):

    SYSTEM_PROMPT_ORIGIN = """
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"name": <function-name>,"arguments": <args-dict>}
</tool_call>

Here are the available tools:
<tools> 
"""

    SYSTEM_PROMPT = """
You are a function calling AI model. You are provided with function signatures within <tools></tools> XML tags. You may call one or more functions to assist with the user query. Don't make assumptions about what values to plug into functions. For each function call return a json object with function name and arguments within <tool_call></tool_call> XML tags as follows:
<tool_call>
{"name": <function-name>,"arguments": <args-dict>}
</tool_call>

** You can and ONLY can use ONE tool **

Here are the available tools:
<tools> 
"""

    @staticmethod
    def get_header_prompt(message) -> str:
        return "<|start_header_id|>" + message["role"] + "<|end_header_id|>\n\n"

    @staticmethod
    def get_message_prompt(message) -> str:
        prompt = GroqChatFormat.get_header_prompt(message)
        prompt += message["content"].strip() + "<|eot_id|>"
        return prompt

    @staticmethod
    def get_dialog_prompt(dialog) -> str:
        prompt = "<|begin_of_text|>"
        for message in dialog:
            prompt += GroqChatFormat.get_message_prompt(message)
        # Add the start of an assistant message for the model to complete.
        prompt += GroqChatFormat.get_header_prompt({"role": "assistant", "content": ""})
        return prompt

    @staticmethod
    def dialog_template(query, tools_with_doc):
        return [
            {
                "role": "system",
                "content": GroqChatFormat.SYSTEM_PROMPT + "\n".join([
                    json.dumps(tool, indent=4) for tool in tools_with_doc
                ]) + "\n</tools>"
            }, {
                "role": "user",
                "content": query
            }
        ]

    @staticmethod
    def post_process(output):
        try:
            output = output.split("<tool_call>")[1].split("</tool_call>")[0]
        except:
            pass
        return output.replace("arguments", "parameters")


class Llama3ChatFormat(ToolChatFormat):

    @staticmethod
    def get_header_prompt(message) -> str:
        return "<|start_header_id|>" + message["role"] + "<|end_header_id|>\n\n"

    @staticmethod
    def get_message_prompt(message) -> str:
        prompt = Llama3ChatFormat.get_header_prompt(message)
        prompt += message["content"].strip() + "<|eot_id|>"
        return prompt

    @staticmethod
    def get_dialog_prompt(dialog) -> str:
        prompt = "<|begin_of_text|>"
        for message in dialog:
            prompt += Llama3ChatFormat.get_message_prompt(message)
        # Add the start of an assistant message for the model to complete.
        prompt += Llama3ChatFormat.get_header_prompt({"role": "assistant", "content": ""})
        return prompt


class Llama2ChatFormat(ToolChatFormat):
    @staticmethod
    def get_dialog_prompt(dialog) -> str:
        B_INST, E_INST = "[INST]", "[/INST]"
        B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
        if dialog[0]["role"] == "system":
            dialog = [
                         {
                             "role": dialog[1]["role"],
                             "content": B_SYS
                                        + dialog[0]["content"]
                                        + E_SYS
                                        + dialog[1]["content"],
                         }
                     ] + dialog[2:]
        dialog_tokens = "".join(
            [
                f"<s>{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} </s>"
                for prompt, answer in zip(
                dialog[::2],
                dialog[1::2],
            )
            ],
        )
        dialog_tokens += f"<s>{B_INST} {(dialog[-1]['content']).strip()} {E_INST}"
        return dialog_tokens


class Llama31ChatFormat(ToolChatFormat):

    SYSTEM_PROMPT = (
        f"\n\nCutting Knowledge Date: December 2023\n"
        f"Today Date: {date.today().strftime('%d %b %Y')}\n\n"
        "When you receive a tool call response, use the output to format an answer to the original user question.\n"
        "You are a helpful assistant with tool calling capabilities."
    )

    USER_PROMPT_1 = """Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

"""

    @staticmethod
    def get_header_prompt(message) -> str:
        return "<|start_header_id|>" + message["role"] + "<|end_header_id|>\n\n"

    @staticmethod
    def get_message_prompt(message) -> str:
        prompt = Llama31ChatFormat.get_header_prompt(message)
        prompt += message["content"].strip() + "<|eot_id|>"
        return prompt

    @staticmethod
    def get_dialog_prompt(dialog) -> str:
        prompt = "<|begin_of_text|>"
        for message in dialog:
            prompt += Llama31ChatFormat.get_message_prompt(message)
        # Add the start of an assistant message for the model to complete.
        prompt += Llama31ChatFormat.get_header_prompt({"role": "assistant", "content": ""})
        return prompt

    @staticmethod
    def dialog_template(query, tools_with_doc):
        return [
            {
                "role": "system",
                "content": Llama31ChatFormat.SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": Llama31ChatFormat.USER_PROMPT_1 + "\n".join([
                    json.dumps({
                            "type": "function",
                            "function": tool
                        }, indent=4)
                    for tool in tools_with_doc
                ]) + "\n\nQuestion: " + query
            }
        ]


class Qwen25ChatFormat(ToolChatFormat):

    SYSTEM_PROMPT_PREFIX = """You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

Current Date: """+date.today().strftime('%Y-%m-%d')+"""

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>"""
    
    SYSTEM_PROMPT_SUFFIX = """</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "parameters": <args-json-object>}
</tool_call><|im_end|>"""

    @staticmethod
    def get_header_prompt(message) -> str:
        return "<|im_start|>" + message["role"] + "\n"

    @staticmethod
    def get_message_prompt(message) -> str:
        prompt = Qwen25ChatFormat.get_header_prompt(message)
        prompt += message["content"].strip() + "<|im_end|>"
        return prompt

    @staticmethod
    def get_dialog_prompt(dialog) -> str:
        prompt = ""
        for message in dialog:
            prompt += Qwen25ChatFormat.get_message_prompt(message)
        # Add the start of an assistant message for the model to complete.
        prompt += Qwen25ChatFormat.get_header_prompt({"role": "assistant", "content": ""})
        return prompt

    @staticmethod
    def dialog_template(query, tools_with_doc):
        return [
            {
                "role": "system",
                "content": Qwen25ChatFormat.SYSTEM_PROMPT_PREFIX + \
                    "\n".join([json.dumps({"type": "function", "function": tool}) for tool in tools_with_doc]) + \
                    Qwen25ChatFormat.SYSTEM_PROMPT_SUFFIX
            },
            {
                "role": "user",
                "content": query
            }
        ]
    
    @staticmethod
    def post_process(output):
        try:
            res = output.split("<tool_call>")[1].split("</tool_call>")[0].replace("arguments", "parameters")
        except:
            res = ""
        return res

    

class Qwen2ChatFormat(ToolChatFormat):

    # Qwen2
    FN_NAME = '✿FUNCTION✿'
    FN_ARGS = '✿ARGS✿'
    FN_RESULT = '✿RESULT✿'
    FN_EXIT = '✿RETURN✿'

    SYSTEM_PROMPT = (
        "You are an expert in invoking function to solve the problem. "
        "You are given a list of functions in JSON format and a question. "
        "Based on the question, you need to make a helpful function call. "
        'You should ONLY give the function call in JSON format like {"name": "API", "parameters":{} }'
    )

    FN_CALL_TEMPLATE_INFO_ZH = """# 工具

    ## 你拥有如下工具：

    {tool_descs}"""

    FN_CALL_TEMPLATE_INFO_EN = """# Tools

    ## You have access to the following tools:

    {tool_descs}"""

    FN_CALL_TEMPLATE_FMT_ZH = """## 你可以在回复中插入零次、一次或多次以下命令以调用工具：

    %s: 工具名称，必须是[{tool_names}]之一。
    %s: 工具输入
    %s: 工具结果
    %s: 根据工具结果进行回复，需将图片用![](url)渲染出来""" % (
        FN_NAME,
        FN_ARGS,
        FN_RESULT,
        FN_EXIT,
    )

    FN_CALL_TEMPLATE_FMT_EN = """## When you need to call a tool, please insert the following command in your reply, which can be called zero or multiple times according to your needs:

    %s: The tool to use, should be one of [{tool_names}]
    %s: The input of the tool
    %s: Tool results
    %s: Reply based on tool results. Images need to be rendered as ![](url)""" % (
        FN_NAME,
        FN_ARGS,
        FN_RESULT,
        FN_EXIT,
    )

    FN_CALL_TEMPLATE = {
        'zh': FN_CALL_TEMPLATE_INFO_ZH + '\n\n' + FN_CALL_TEMPLATE_FMT_ZH,
        'en': FN_CALL_TEMPLATE_INFO_EN + '\n\n' + FN_CALL_TEMPLATE_FMT_EN
    }

    @staticmethod
    def get_header_prompt(message) -> str:
        return "<|im_start|>" + message["role"] + "\n"

    @staticmethod
    def get_message_prompt(message) -> str:
        prompt = Qwen2ChatFormat.get_header_prompt(message)
        prompt += message["content"].strip() + "<|im_end|>"
        return prompt

    @staticmethod
    def get_dialog_prompt(dialog) -> str:
        prompt = ""
        for message in dialog:
            prompt += Qwen2ChatFormat.get_message_prompt(message)
        # Add the start of an assistant message for the model to complete.
        prompt += Qwen2ChatFormat.get_header_prompt({"role": "assistant", "content": ""})
        return prompt

    @staticmethod
    def get_function_description(function, lang='en') -> str:
        """
        Text description of function
        """
        tool_desc_template = {
            'zh': '### {name_for_human}\n\n{name_for_model}: {description_for_model} 输入参数：{parameters} {args_format}',
            'en': '### {name_for_human}\n\n{name_for_model}: {description_for_model} Parameters: {parameters} {args_format}'
        }
        tool_desc = tool_desc_template[lang]
        name = function.get('name', None)
        name_for_human = function.get('name_for_human', name)
        name_for_model = function.get('name_for_model', name)
        assert name_for_human and name_for_model

        if name_for_model == 'code_interpreter':
            args_format = {
                'zh': '此工具的输入应为Markdown代码块。',
                'en': 'Enclose the code within triple backticks (`) at the beginning and end of the code.',
            }
        else:
            args_format = {
                'zh': '此工具的输入应为JSON对象。',
                'en': 'Format the arguments as a JSON object.',
            }
        args_format = function.get('args_format', args_format[lang])

        return tool_desc.format(
            name_for_human=name_for_human,
            name_for_model=name_for_model,
            description_for_model=function['description'],
            parameters=json.dumps(function['parameters'], ensure_ascii=False),
            args_format=args_format
        ).rstrip()

    @staticmethod
    def dialog_template(query, tools_with_doc, lang="en"):
        """
        这是我们编写的 llama2 和 llama3 使用工具的模板（因为官方没有提供模板）
        你需要参考它们和其它模型的官方文档，实现其它模型调用工具的模板
        """
        # Qwen2：往system prompt中加入工具信息
        functions = tools_with_doc
        tool_desc_template = Qwen2ChatFormat.FN_CALL_TEMPLATE[lang]
        tool_descs = '\n\n'.join(
            Qwen2ChatFormat.get_function_description(function, lang=lang) for function in functions
        )
        tool_names = ','.join(function.get('name', function.get('name_for_model', '')) for function in functions)
        tool_system = tool_desc_template.format(tool_descs=tool_descs, tool_names=tool_names)

        return [
            {
                "role": "system",
                "content": Qwen2ChatFormat.SYSTEM_PROMPT + '\n\n' + tool_system
            },
            {
                "role": "user",
                "content": query
            }
        ]

    @staticmethod
    def post_process(output):
        output_lines = output.split("\n")
        name = ""
        parameters = ""
        for line in output_lines:
            if Qwen2ChatFormat.FN_NAME in line:
                name = ":".join(line.split(":")[1:]).strip()
            if Qwen2ChatFormat.FN_ARGS in line:
                parameters = ":".join(line.split(":")[1:]).strip()
        return "{" + f'"name": "{name}", "parameters": {parameters}' + "}"


class GLM4ChatFormat(ToolChatFormat):

    SYSTEM_PROMPT = """你是一个名为 GLM-4 的人工智能助手。你是基于智谱AI训练的语言模型 GLM-4 模型开发的，你的任务是针对用户的问题和要求提供适当的答复和支持。

# 可用工具
## get_recommended_books

{}
在调用上述函数时，请使用 Json 格式表示调用的参数。
"""

    USER_PROMPT_1 = """Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

"""

    @staticmethod
    def get_header_prompt(message) -> str:
        return "<|" + message["role"] + "|>\n"

    @staticmethod
    def get_message_prompt(message) -> str:
        prompt = GLM4ChatFormat.get_header_prompt(message)
        prompt += message["content"].strip()
        return prompt

    @staticmethod
    def get_dialog_prompt(dialog) -> str:
        prompt = "[gMASK]<sop>"
        for message in dialog:
            prompt += GLM4ChatFormat.get_message_prompt(message)
        # Add the start of an assistant message for the model to complete.
        prompt += GLM4ChatFormat.get_header_prompt({"role": "assistant", "content": ""})
        return prompt

    @staticmethod
    def dialog_template(query, tools_with_doc):
        return [
            {
                "role": "system",
                "content": GLM4ChatFormat.SYSTEM_PROMPT.format("\n".join([
                    json.dumps(tool, indent=4) for tool in tools_with_doc
                ]))
            },
            {
                "role": "user",
                "content": query
            }
        ]

    @staticmethod
    def post_process(output):
        output = output.replace("arguments", "parameters")
        for response in output.split("<|assistant|>"):
            if "\n" in response:
                metadata, content = response.split("\n", maxsplit=1)
            else:
                metadata, content = "", response
            if metadata.strip():
                return '{' + f'"name": "{metadata.strip()}", "parameters": {content}' + '}'
        return "No Function Call"
