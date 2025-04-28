import os
import re
import sys
import json
import ast

def construct_answer(answer, list_of_roles):
    content = []

    for call in answer:
        name = call
        if answer[call]:
            try:
                parameters = answer[call]
                if isinstance(parameters, dict):
                    pass
                elif isinstance(parameters, str):
                    parameters = parameters.replace(f"true", f"True")
                    parameters = eval(parameters)
            except Exception as e:
                print(e)
            depend_on = []

            for param in parameters:
                try:
                    if isinstance(parameters[param], str):
                        matched = re.match(r'([^.]+)\.(.+)', parameters[param])
                        if matched:
                            depended_name = matched.group(1)
                            depended_parameter = matched.group(2)
                            depended_calls = []
                            for role in list_of_roles:
                                if role['role']=='tool_call':
                                    depended_calls = depended_calls + [call for call in role['content'] if call['name'] == depended_name]
                            depended_calls = depended_calls + [call for call in content if call['name'] == depended_name]
                            if not depended_calls:
                                for role in list_of_roles:
                                    if role['role']=='candidate_tools':
                                        for tool in role['content']:
                                            if tool['name'] == depended_name:
                                                added_name = depended_name
                                                added_parameters = {}
                                                for depended_param in tool['parameters']['properties']:
                                                    if depended_param in tool['parameters']['required']:
                                                        if tool['parameters']['properties'][depended_param]:
                                                            added_parameters[depended_param] = tool['parameters']['properties'][depended_param]['default']
                                                added_content = {
                                                    "name": added_name,
                                                    "parameters": added_parameters,
                                                    "depend_on": [],
                                                }
                                                content.append(added_content)
                                                break
                                        break
                                depended_calls = depended_calls + [call for call in content if call['name'] == depended_name]
                            if not depended_calls:
                                break
                            depended_call = depended_calls[-1]
                            depended_call_number = len(depended_calls)-1
                            parameters[param] = f"<link>{depended_name}.{depended_call_number}.{depended_parameter}</link>"
                            if f"{depended_name}.{depended_call_number}" not in depend_on:
                                depend_on.append(f"{depended_name}.{depended_call_number}")
                except Exception as e:
                    print(f" [ERROR]{list_of_roles[0]['content']}:{e}")
            once_content = {
                "name": name,
                "parameters": parameters,
                "depend_on": depend_on,
            }
            content.append(once_content)

    to_write = {
        "role": "tool_call",
        "content": content,
    }
    list_of_roles.append(to_write)


def construct_candidate_tools_for_ood(init_tools):

    available_parameters = init_tools["available_parameters"]
    apis = init_tools["apis"]

    content = []

    parameters = {
        "properties" : available_parameters,
        "required" : []
    }

    for api in apis:
        one_content = {
            "name": api["name"],
            "description": api["description"],
            "parameters": parameters,
        }
        content.append(one_content)


    candidate_tools = {
        "role": "candidate_tools",
        "content": content,
    }
    return candidate_tools




def construct_candidate_tools(question, id):
    lines = question.splitlines()
    ptr = 0
    for line in lines:
        if ptr == 1:
            if line != "":
                extracted_text = line
                break
        if line.startswith('The following is a list of APIs and their parameters that you can use:'):
            matched = re.match('The following is a list of APIs and their parameters that you can use:(.+)', line)
            if matched:
                extracted_text = matched.group(1)
                break
            else:
                ptr = 1
    init_tools = ast.literal_eval(extracted_text)
    content = []
    for tool in init_tools:
        try:
            if isinstance(tool, dict):
                # print(f"{tool} is dict")
                name = tool['name']
                description = tool['description']
            elif isinstance(tool, str):
                return construct_candidate_tools_for_ood(init_tools)
            else:
                raise Exception(f"init_tool is neither a dict nor a string")
        except Exception as e:
            print(f"{id} error: {e}, with tool:{tool}")
            return {
                "role": "candidate_tools",
                "content": [],
            }

        properties = {}
        for param in tool['optional_parameters']:
            if isinstance(tool['optional_parameters'], dict):
                properties[param] = {
                    "default" : tool['optional_parameters'][param]
                }
            if isinstance(tool['optional_parameters'], list):
                properties[param['name']] = param
                del properties[param['name']]['name']
        required_parameters = tool['required_parameters']
        required = []
        if required_parameters:
            for param in required_parameters:
                if isinstance(param, str):
                    required.append(param)
                    if param not in properties:
                        properties[param] = {
                            'description': '',
                            'default': '',
                        }
                elif isinstance(param, dict):
                    properties[param['name']] = param
                    required.append(param['name'])
                    del properties[param['name']]['name']


        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        response = {}
        for result in tool['result_parameters']:
            if isinstance(result, str):
                response[result] = {}
            elif isinstance(result, dict):
                response[result['name']] = result
                del response[result['name']]['name']
            else:
                raise Exception("tool['result_parameters'] error")

        one_content = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "response": response,
        }
        content.append(one_content)


    candidate_tools = {
        "role": "candidate_tools",
        "content": content,
    }
    return candidate_tools

def query_for_user(list_of_queries, list_of_roles, role, content):
    list_of_queries.pop(0)
    role_to_append = {
        "role": "user",
        "content": content,
    }
    list_of_roles.append(role_to_append)

def query_for_function(list_of_queries, list_of_roles, role, content):
    list_of_queries.pop(0)


def query_for_thought(list_of_queries, list_of_roles, role, content):
    list_of_queries.pop(0)
    role_to_append = {
        "role": "assistant",
        "hidden":True,
        "content": content,
    }
    list_of_roles.append(role_to_append)


def query_for_action(list_of_queries, list_of_roles, role, content):
    if content!="{}":
        answer = {}
        for line, next_line in zip(list_of_queries, list_of_queries[1:]):
            if line.startswith("Action") :
                if next_line.startswith("Action Input"):
                    matched_in_line = re.match(f"Action:(.+)", list_of_queries.pop(0))
                    matched_in_next_line = re.match(f"Action Input:(.+)", list_of_queries.pop(0))

                    name = matched_in_line.group(1)
                    parameters = matched_in_next_line.group(1)
                    answer[name] = parameters

                else:
                    try:
                        matched_in_line = re.match(f"Action:(.+)", list_of_queries.pop(0))
                        if matched_in_line:
                            calls = matched_in_line.group(1)
                            calls = calls.replace(f"true", f"True")
                            answer = eval(calls)
                        else:
                            answer = {}
                    except Exception as e:
                        answer = {}
            else:
                break
        construct_answer(answer, list_of_roles)
    else:
        list_of_queries.pop(0)



def query_for_observation(list_of_queries, list_of_roles, role, observation):
    role = "tool_response"
    content = {}

    list_of_tool_call = [ role for role in list_of_roles if role["role"] == "tool_call"]
    if list_of_tool_call and observation:
        called_tool = list_of_tool_call[-1]["content"][0]
        matched_tools_number = 0
        for once_call in list_of_tool_call:
            for one_tool in once_call['content']:
                if one_tool["name"] == called_tool["name"]:
                    matched_tools_number += 1
        name = called_tool["name"]+f".{matched_tools_number-1}"
        new_observation = observation.replace(f"true", f"True")
        new_observation = new_observation.replace(f"...", f"")
        while ']]' in new_observation:
            new_observation = re.sub(r'\]\]', ']', new_observation)
        try:
            content[name] = eval(new_observation)
        except Exception as e:
            print(f"line 253: ",e)

    role_to_append = {
        "role": "tool_response",
        "content": content,
    }
    list_of_roles.append(role_to_append)
    list_of_queries.pop(0)


def query_for_assistant(list_of_queries, list_of_roles, role, content):
    list_of_queries.pop(0)
    role_to_append = {
        "role": "assistant",
        "content": content,
    }
    list_of_roles.append(role_to_append)

def construct_query(question, list_of_roles):
    lines = question.splitlines()
    ptr = 0
    init_query = []
    for line in lines:
        if ptr == 2:
            if line != "":
                init_query.append(line)
        if ptr == 1:
            if line != "":
                ptr = 2
        if line.startswith('The following is a list of APIs'):
            matched = re.match('The following is a list of APIs and their parameters that you can use:(.+)', line)
            if matched:
                ptr = 2
            else:
                ptr = 1

    while init_query:
        query = init_query[0]
        if query.startswith('History'):
            init_query.pop(0)
            continue
        else:
            try:
                matched = re.match(r"([^:]+)([:])([\s]*)([:]*)(.+)", query)
                if matched:
                    role = matched.group(1)
                    content = matched.group(5)
                    if role == "user" or role == "User":
                        query_for_user(init_query, list_of_roles, role, content)
                    elif role == "function" or role == "Function":
                        query_for_function(init_query, list_of_roles, role, content)
                    elif role == "thought" or role == "Thought":
                        query_for_thought(init_query, list_of_roles, role, content)
                    elif role == "action" or role == "Action":
                        query_for_action(init_query, list_of_roles, role, content)
                    elif role == "observation" or role == "Observation":
                        if list_of_roles[-1]["role"] == "tool_call":
                            query_for_observation(init_query, list_of_roles, role, content.strip())
                        else:
                            # query_for_thought(init_query, list_of_roles, role, content.strip())
                            init_query.pop(0)
                    elif role == "assistant" or role == "Assistant":
                        query_for_assistant(init_query, list_of_roles, role, content)
                else:
                    init_query.pop(0)
            except Exception as e:
                print(f"line 254: ",e)


def process_s_s(from_path, to_path):
    tools_list = []

    with open(os.path.join(from_path, "S-S_eval.jsonl")) as fin:
        to_dump = []

        for line in fin:

            # initial
            id = json.loads(line)["id"]
            question = json.loads(line)["question"]
            answer = json.loads(line)["answer"]
            list_of_roles = []
            list_of_roles.append({"role": "id", "content": "MTU-Bench_"+id})

            # "role":"candidate_tools"
            candidate_tools = construct_candidate_tools(question, id)
            tools_list.append(candidate_tools)
            list_of_roles.append(candidate_tools)

            # query
            construct_query(question, list_of_roles)

            # answer
            construct_answer(answer, list_of_roles)

            to_dump.append(list_of_roles)

        # dump
        with open(os.path.join(to_path, "S-S.jsonl"), "w") as fout:
            fout.write("\n".join([json.dumps(value) for value in to_dump]))

    print("S_S processed!")
    return tools_list

def process_s_m(from_path, to_path):
    tools_list = []
    to_dump = []

    with open(os.path.join(from_path, "S-M_eval.jsonl")) as fin:
        for line in fin:
            # initial
            id = json.loads(line)["id"]
            question = json.loads(line)["question"]
            answer = json.loads(line)["answer"]
            list_of_roles = []
            list_of_roles.append({"role": "id", "content": "MTU-Bench_"+id})

            # "role":"candidate_tools"
            candidate_tools = construct_candidate_tools(question, id)
            tools_list.append(candidate_tools)
            list_of_roles.append(candidate_tools)

            # query
            construct_query(question, list_of_roles)

            # answer
            construct_answer(answer, list_of_roles)

            to_dump.append(list_of_roles)

        # dump
        with open(os.path.join(to_path, "S-M.jsonl"), "w") as fout:
            fout.write("\n".join([json.dumps(value) for value in to_dump]))
    print("S_M processed!")
    return tools_list


def process_m_s(from_path, to_path):
    tools_list = []
    with open(os.path.join(from_path, "M-S_eval.jsonl")) as fin:
        lines = fin.readlines()
    def mycmpforlines(line):
        matched = re.match("M-S_([^_]+)_(.+)", json.loads(line)["id"])
        group = int(matched.group(1))
        number = int(matched.group(2))
        return group*1000+number
    lines.sort(key=mycmpforlines)


    to_dump = []
    for line, nextline in zip(lines,lines[1:]):
        # initial
        id = re.match("(M-S_[^_]+)", json.loads(line)["id"]).group(1)
        next_id = re.match("(M-S_[^_]+)", json.loads(nextline)["id"]).group(1)
        if id == next_id:
            continue
        question = json.loads(line)["question"]
        answer = json.loads(line)["answer"]
        list_of_roles = []
        list_of_roles.append({"role": "id", "content": "MTU-Bench_"+id})

        # "role":"candidate_tools"
        candidate_tools = construct_candidate_tools(question, id)
        tools_list.append(candidate_tools)
        list_of_roles.append(candidate_tools)

        # query
        construct_query(question, list_of_roles)

        # answer
        construct_answer(answer, list_of_roles)

        to_dump.append(list_of_roles)

    # dump
    with open(os.path.join(to_path, "M-S.jsonl"), "w") as fout:
        try:
            fout.write("\n".join([json.dumps(value) for value in to_dump]))
        except Exception as e:
            print("Error when dumping: ",e)
    print("M_S processed!")
    return tools_list


def process_m_m(from_path, to_path):
    tools_list = []
    with open(os.path.join(from_path, "M-M_eval.jsonl")) as fin:
        lines = fin.readlines()

    to_dump = []
    for line, nextline in zip(lines,lines[1:]):
        # initial
        id = re.match("(M-M_[^_]+)", json.loads(line)["id"]).group(1)
        next_id = re.match("(M-M_[^_]+)", json.loads(nextline)["id"]).group(1)
        if id == next_id:
            continue
        question = json.loads(line)["question"]
        answer = json.loads(line)["answer"]
        list_of_roles = []
        list_of_roles.append({"role": "id", "content": "MTU-Bench_"+id})

        # "role":"candidate_tools"
        candidate_tools = construct_candidate_tools(question, id)
        tools_list.append(candidate_tools)
        list_of_roles.append(candidate_tools)

        # query
        construct_query(question, list_of_roles)

        # answer
        construct_answer(answer, list_of_roles)

        to_dump.append(list_of_roles)

    # dump
    with open(os.path.join(to_path, "M-M.jsonl"), "w") as fout:
        fout.write("\n".join([json.dumps(value) for value in to_dump]))
                
    print("M_M processed!")
    return tools_list


def process_ood(from_path, to_path):
    tools_list = []
    with open(os.path.join(from_path, "OOD_eval.jsonl")) as fin:
        to_dump = []
        for line in fin:
            # initial
            id = json.loads(line)["id"]
            question = json.loads(line)["question"]
            answer = json.loads(line)["answer"]
            list_of_roles = []
            list_of_roles.append({"role": "id", "content": "MTU-Bench_"+id})

            # "role":"candidate_tools"
            candidate_tools = construct_candidate_tools(question, id)
            tools_list.append(candidate_tools)
            list_of_roles.append(candidate_tools)

            # query
            construct_query(question, list_of_roles)

            # answer
            construct_answer(answer, list_of_roles)

            to_dump.append(list_of_roles)

        # dump
        with open(os.path.join(to_path, "OOD.jsonl"), "w") as fout:
            fout.write("\n".join([json.dumps(value) for value in to_dump]))
    print("OOD processed!")
    return tools_list


def process_mtu_bench(from_path, to_path, tool_path):
    # print("warning: MTU-Bench processor not done yet!")
    tools_list = []
    tools_list.extend(process_s_s(from_path, to_path))
    tools_list.extend(process_s_m(from_path, to_path))
    tools_list.extend(process_m_s(from_path, to_path))
    tools_list.extend(process_m_m(from_path, to_path))
    tools_list.extend(process_ood(from_path, to_path))
    print("MTU-Bench processed!")

    tool_set = set()
    for tools in tools_list:
        for tool in tools["content"]:
            tool_set.add(json.dumps(tool, ensure_ascii=False))
    with open(os.path.join(tool_path, "tools_with_doc.jsonl"), "w") as fout:
        fout.write("\n".join([tool for tool in tool_set]))
        print(os.path.abspath(os.path.join(tool_path, "tools_with_doc.jsonl")), "saved.")


if __name__ == "__main__":
    process_mtu_bench()