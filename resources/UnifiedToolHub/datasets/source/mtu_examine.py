import os
import re
import sys
import json
import ast
from mtu_bench import TO_PATH

FROM_PATH = os.path.join(os.path.dirname(__file__), "downloaded", "MTU-Bench")
# TO_PATH = os.path.join(os.path.dirname(__file__), "processed", "MTU-Bench")
LIST_OF_FILES = ["S-S.jsonl", "S-M.jsonl", "M-S.jsonl", "M-M.jsonl", "OOD.jsonl"]



class Error(object):
    def __init__(self):
        self.__number = 0
        self.__message = []

    def add_message(self, message):
        self.__message.append(message)
    def add_number(self):
        self.__number += 1

    def number(self):
        return self.__number

    def message(self):
        return self.__message

    def print(self):
        for message in self.__message:
            print("\t", message)

    def str(self):
        return str(self.__message)

class Examine(object):
    def __init__(self, passages):
        self.__error = Error()

        if not isinstance(passages, list):
            self.__error.add_number()
            self.__error.add_message("The passages must be a list!")
        for passage in passages:
            if not isinstance(passage, dict):
                self.__error.add_number()
                self.__error.add_message("The passage must be a dict!")
            role = passage["role"]
            context = passage["context"]
            self.__examine_role(role, context)

    def error(self):
        return self.__error

    def __examine_role(self, role, context):
        if role == "id":
            self.__examine_id(context)
        elif role == "candidate_tools":
            self.__examine_candidate_tools(context)
        elif role == "user":
            self.__examine_user(context)
        elif role == "assistant":
            self.__examine_assistant(context)
        elif role == "tool_call":
            self.__examine_tool_call(context)
        elif role == "tool_response":
            self.__examine_tool_response(context)

    def __examine_id(self, context):
        if not isinstance(context, str):
            self.__error.add_number()
            self.__error.add_message("The <id> must be a string!")

    def __examine_candidate_tools(self, context):
        if not isinstance(context, list):
            self.__error.add_number()
            self.__error.add_message("The <candidate_tools> must be a list!")
        if not context:
            self.__error.add_number()
            self.__error.add_message("The <candidate_tools> is empty!")
        for tool in context:
            if not isinstance(tool, dict):
                self.__error.add_number()
                self.__error.add_message("The <tool> must be a dict!")
            if not "name" in tool:
                self.__error.add_number()
                self.__error.add_message(f"The <tool> have no \"name\"!")
            if not "parameters" in tool:
                self.__error.add_number()
                self.__error.add_message("The <tool> have no \"parameters\"!")

    def __examine_user(self, context):
        if not isinstance(context, str):
            self.__error.add_number()
            self.__error.add_message("The <user> must be a string!")

    def __examine_assistant(self, context):
        if not isinstance(context, str):
            self.__error.add_number()
            self.__error.add_message("The <assistant> must be a string!")

    def __examine_tool_call(self, context):
        if not isinstance(context, list):
            self.__error.add_number()
            self.__error.add_message("The <tool_call> must be a list!")

    def __examine_tool_response(self, context):
        if not isinstance(context, dict):
            self.__error.add_number()
            self.__error.add_message("The <tool_response> must be a dict!")


def examine_lines(lines):
    list_to_write = []

    for line in lines:
        list_of_passages = json.loads(line)
        id = list_of_passages[0]["context"]
        # error = 0

        examine = Examine(list_of_passages)
        error = examine.error()
        del examine

        if error.number():
            print(id, "is wrong: ")
            error.print()
            to_write = {
                "line": list_of_passages,
                "error": error.str(),
            }
            list_to_write.append(to_write)

    return list_to_write


def examine_mtu_bench():
    for filename in LIST_OF_FILES:

        with open(os.path.join(TO_PATH, filename), "r", encoding="utf-8") as fin:
            lines = fin.readlines()

        output = examine_lines(lines)

        with open(os.path.join("output", filename), "w", encoding="utf-8") as fout:
            fout.write("\n".join([json.dumps(value, indent=4) for value in output]))


if __name__ == "__main__":
    examine_mtu_bench()