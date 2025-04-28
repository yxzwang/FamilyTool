import ast
import json
import os
import re
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

import requests
from openai import OpenAI

__BASE_DIR__ = os.path.dirname(os.path.abspath(__file__))

# 以下是一些通用的组件

class Requester:
    def __init__(self, base_url):
        self.client = OpenAI(
            api_key="EMPTY",
            base_url=base_url,
        )
        self.model = self.client.models.list().data[0].id

    def chat(self, messages: list, **kwargs):
        params = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 4096,
        }
        params = {
            **params,
            **kwargs
        }
        result = self.client.chat.completions.create(**params)
        return result


def offline_tagger(model, sampling_params, data_list, preprocess_func, postprocess_func):   
    chats = [preprocess_func(data) for data in data_list] 
    res_list = model.chat(chats, sampling_params)
    return [
        postprocess_func(data, res.outputs[0].text) 
        for data, res in zip(data_list, res_list)
    ]

def online_reqester_tagger(requester_idx, data, chat):
    try:
        res = global_requester_list[requester_idx].chat(chat, **global_sampling_params)
        return res.choices[0].message.content
    except Exception as e:
        print(e)
        return "Error"
        

def online_tagger(requester_list, sampling_params, data_list, preprocess_func, postprocess_func, processes_num=mp.cpu_count()):
    T = len(requester_list)
    global global_requester_list, global_sampling_params
    global_requester_list = requester_list
    global_sampling_params = sampling_params

    params = [
        (idx%T, data, preprocess_func(data)) 
        for idx, data in enumerate(data_list)
    ]

    with Pool(processes=processes_num) as pool:
        results = pool.starmap(online_reqester_tagger, params)
    
    ret_list = []
    for data, res in zip(data_list, results):
        # 此处可以对出现错误的再进行处理
        ret_list.append(postprocess_func(data, res))

    return ret_list

# 以下是一个示例 Demo-RapidAPI-tag
       
SYSTEM_PROMPT = """
You are an intelligent assistant designed to analyze tools and classify them into specific categories based on their functionalities, purposes, and characteristics. Your task is to evaluate the provided tool description and determine if it fits into any of the specified tool categories.

Tool Categories:

- Financial : Financial APIs link users to various financial institutions and news outlets, enabling access to account information, market rates, and industry news. They facilitate trading and decision-making, enhancing efficiency in financial transactions.
- Communication : A Communication API enables developers to integrate text messaging, voice calling, and other communication functionalities into their applications or for businesses. These APIs include voice, SMS, MMS, RCS, chat, and emergency calling.
- Jobs : Jobs APIs provide access to job-related data, such as job listings, career opportunities, company profiles, and employment statistics.
- Music : Music APIs enable developers to integrate music and its associated data into various applications and services, offering functionalities such as streaming, displaying lyrics, and providing metadata like song details and artist information.
- Travel : Travel APIs provide real-time information on hotel prices, airline itineraries, and destination recommendations.
- Social : Social APIs enable developers to integrate social media platforms into their applications, allowing for connectivity and access to platform databases for analytical or manipulation purposes.
- Sports : Sports APIs encompass various categories such as sports odds, top scores, NCAA, football, women's sports, and trending sports news.
- Database : A Database API facilitates communication between applications and databases, retrieving requested information stored on servers.
- Finance : Finance APIs offer users diverse services for account management and staying informed about market events.
- Data : APIs facilitate the seamless exchange of data between applications and databases.
- Food : Food APIs link users' devices to vast food-related databases, offering features like recipes, nutritional information, and food delivery services.
- Entertainment : Entertainment APIs range from movies and love interest research to jokes, memes, games, and music exploration.
- Text_Analysis : Text Analysis APIs leverage AI and NLP to dissect large bodies of text, offering functionalities such as translation, fact extraction, sentiment analysis, and keyword research.
- Translation : Translation APIs integrate cloud translation services into applications, facilitating text translation between applications and web pages.
- Location : Location APIs power applications that depend on user location for relevant results.
- Business_Software : Business software APIs streamline communication between different business applications.
- Movies : Movie APIs connect applications or websites to servers housing movie-related information or files.
- Business : Business APIs cover a wide range of functionalities, from e-commerce inventory management to internal operations and customer-facing interactions.
- Science : Science APIs facilitate access to a plethora of scientific knowledge.
- eCommerce : Email APIs enable users to access and utilize the functionalities of email service providers.
- Monitoring : A Monitoring API enables applications to access data for tracking various activities.
- Tools : Tool APIs offer a diverse range of functionalities, from text analysis to generating QR codes and providing chatbot services.
- Transportation : Transportation APIs connect users to transit system databases.
- Email : Email APIs enable users to access and utilize the functionalities of email service providers.
- Mapping : Mapping APIs provide location services and intelligence to developers for various applications.
- Gaming : Gaming APIs connect users to game servers for tasks like account management and gameplay analysis.
- Search : Search APIs allow developers to integrate search functionality into their applications or websites.
- Health_and_Fitness : Health and fitness APIs offer tools for managing nutrition, exercise, and health monitoring.
- Weather : Weather APIs provide users with access to accurate forecasts and meteorological data.
- Education : Education APIs facilitate seamless access to educational resources.
- News_Media : News and Media APIs allow developers to integrate news and media content into their applications.
- Reward : Reward APIs simplify the implementation of rewards and coupon systems into applications.
- Others : This category includes any tools or documents that do not fit into the above classifications.
""".strip()

USER_PROMPT_TEMPLE = """
Please analyze the following tool document and determine whether it belongs to any of the above categories.

Tool Document: {}

Please output the category that the tool belongs to. If it does not fit into any category, please indicate "Others". You should only output ONE WORD as the category. 
""".strip()

RAPIDAPI_TAGS = set(["Financial", "Communication", "Jobs", "Music", "Travel", "Social", "Sports", "Database", "Finance", "Data", "Food", "Entertainment", "Text_Analysis", "Translation", "Location", "Business_Software", "Movies", "Business", "Science", "eCommerce", "Monitoring", "Tools", "Transportation", "Email", "Mapping", "Gaming", "Search", "Health_and_Fitness", "Weather", "Education", "News_Media", "Reward", "Others"])


def preprocess_rapidapi_tag(data):
    return [{
        "role": "system",
        "content": SYSTEM_PROMPT
    }, {
        "role": "user",
        "content": USER_PROMPT_TEMPLE.format(json.dumps(data["tool"]))
    }]


def postprocess_rapidapi_tag(data, res):
    if res not in RAPIDAPI_TAGS:
        res ='Others'
    return json.dumps({
        "id": data["id"],
        "tag": res
    })


def demo_get_rapidapi_online_and_save_to_file():
    requester_list = [
        Requester(base_url="http://10.80.3.162:8000/v1"),
    ]
    sampling_params = {"max_tokens": 4096}
    demo_data_list = [{
        "id": "bfcl_v3_1",
        "tool": {"name": "physics.calculate_force", "description": "Calculate the force required to move an object of a particular mass at a particular acceleration.", "parameters": {"type": "dict", "properties": {"mass": {"type": "integer", "description": "The mass of the object in kg."}, "acceleration": {"type": "integer", "description": "The acceleration of the object in m/s^2."}}, "required": ["mass", "acceleration"]}}
    }]*10

    res_list = online_tagger(
        requester_list,
        sampling_params,
        demo_data_list,
        preprocess_rapidapi_tag,
        postprocess_rapidapi_tag,
        2
    )

    with open(os.path.join(__BASE_DIR__, "rapidapi", "demo_tag.jsonl"), "w") as fout:
        fout.write("\n".join(res_list))

def demo_get_rapidapi_offline(data_list, from_idx, to_idx, save_step=-1, model="Meta-Llama-3.1-405B-Instruct-FP8"):
    from vllm import LLM, SamplingParams

    demo_data_list = [{
        "id": "bfcl_v3_1",
        "tool": {"name": "physics.calculate_force", "description": "Calculate the force required to move an object of a particular mass at a particular acceleration.", "parameters": {"type": "dict", "properties": {"mass": {"type": "integer", "description": "The mass of the object in kg."}, "acceleration": {"type": "integer", "description": "The acceleration of the object in m/s^2."}}, "required": ["mass", "acceleration"]}}
    }]*1000

    if model == "Meta-Llama-3.1-405B-Instruct-FP8":
        model = LLM(
            model="/opt/local/data/Meta-Llama-3.1-405B-Instruct-FP8",
            tensor_parallel_size=8,
            max_num_seqs=25, 
            max_model_len=8192,
            enforce_eager=True,
        )
    elif model == "Qwen2.5-72B-Instruct":        
        model = LLM(
            model="/opt/local/data/Qwen2.5-72B-Instruct",
            tensor_parallel_size=8,
            enforce_eager=True,
        )
    else: # elif model == "Qwen2.5-7B-Instruct":
        model = LLM(
            model="/opt/local/data/Qwen2.5-7B-Instruct",
            enforce_eager=True,
        )
    
    sampling_params = SamplingParams(
        skip_special_tokens=False,
        max_tokens=8192, 
        n=1, 
    )

    if save_step == -1:
        save_step = to_idx - from_idx + 1

    append_path = os.path.join(__BASE_DIR__, "rapidapi", "demo_tag_{}_{}.jsonl".format(from_idx, to_idx))
    
    for i in range(from_idx, to_idx, save_step):
        print("Tagging Dataset-[{},{}) by {}".format(i, i+save_step, model))
        res_list = offline_tagger(
            model,
            sampling_params,
            demo_data_list[i:i+save_step],
            preprocess_rapidapi_tag,
            postprocess_rapidapi_tag
        )
        with open(append_path, "a") as fout:
            fout.write("\n".join(res_list)+"\n")
