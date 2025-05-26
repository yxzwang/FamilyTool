import gc
import json
import os
import re
import traceback
import faiss
from openai import OpenAI
import torch
from transformers import AutoTokenizer
from vllm import LLM
from collections import defaultdict
import gc
import re
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from tqdm import tqdm
from transformers import AutoModel
from datetime import datetime
import inspect
import time
from urllib import request
from urllib.parse import urlencode
WAITING_TIME = 0.5
MAX_OPS = 500  
def clean_links(linkstr:str):
    return linkstr.replace("\'","").replace("\"","")
def extract_links_func(inputstring):
    inputstring=clean_links(inputstring)
    # match = re.findall(r'\[(.*?)\]', inputstring)
    match=re.findall(r'\[([^\[\]]+)\]',inputstring)
    try:
        result_list=[]
        for content in match:
            newcontent=[item.strip() for item in content.split(",")]
            result_list.append(newcontent)
    except:
        import ipdb
        ipdb.set_trace()
    return result_list
def preprocessdata(inputdatas):
    newdata=[]
    for data in inputdatas:
        inputdict=data[2] ##### "role": "user"的dict
        goldquery=inputdict["content"]
        inputdict["query"]=goldquery.split("The extra information for the query")[0].strip()
        assert len(inputdict["query"])>10
        newdata.append(inputdict)
    return newdata
def read_json(file_path):
    """
    读取标准 JSON 文件，返回数据对象
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # 解析 JSON 文件并将其转换为 Python 对象（如字典或列表）
    return data
class LarkReport:

    def __init__(self, app_id, app_secret, bitable_url=None, show_info=True, *args, **kwargs):
        self.storage = {}
        self.APP_ID = app_id
        self.APP_SECRET = app_secret
        self.show_info = show_info
        self._bitable_dict = {}
        if bitable_url:
            self.add_bitable("default", bitable_url)

    def _post_req(self, url, headers, req_body, param=None):
        if param is not None:
            url = url + '?' + urlencode(param)
        if self.show_info:
            print("Post [{}]".format(url))
        data = bytes(json.dumps(req_body), encoding='utf8')
        req = request.Request(url=url, data=data, headers=headers, method='POST')
        try:
            response = request.urlopen(req)
        except Exception as e:
            print("Error", e.read().decode())
            return {}

        rsp_body = response.read().decode('utf-8')
        rsp_dict = json.loads(rsp_body)
        return rsp_dict

    def _get_req(self, url, headers, param=None, method='GET'):
        if param is not None:
            url = url + '?' + urlencode(param)
        if self.show_info:
            print("Get [{}]".format(url))
        req = request.Request(url=url, headers=headers, method=method)
        try:
            response = request.urlopen(req)
        except Exception as e:
            print("Error", e.read().decode())
            return {}

        rsp_body = response.read().decode('utf-8')
        rsp_dict = json.loads(rsp_body)
        return rsp_dict

    def _get_tenant_access_token(self):
        url = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal/"
        headers = {
            "Content-Type": "application/json"
        }
        req_body = {
            "app_id": self.APP_ID,
            "app_secret": self.APP_SECRET
        }

        rsp_dict = self._post_req(url, headers, req_body)

        code = rsp_dict.get("code", -1)
        if code != 0:
            print("get tenant_access_token error, code =", code)
            return ""

        return rsp_dict.get("tenant_access_token", "")

    def post_req(self, url: str, headers: dict = None, req_body: dict = None, param=None):
        if headers is None:
            headers = {}
        if req_body is None:
            req_body = {}
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json; charset=utf-8"
        if "Authorization" not in headers:
            headers["Authorization"] = "Bearer " + self.tenant_access_token()
        return self._post_req(url, headers, req_body, param)

    def delete_req(self, url: str, headers: dict = None, param=None):
        if headers is None:
            headers = {}
        if "Content-Type" not in headers:
            headers["Content-Type"] = "application/json; charset=utf-8"
        if "Authorization" not in headers:
            headers["Authorization"] = "Bearer " + self.tenant_access_token()
        return self._get_req(url, headers, param, method="DELETE")

    def get_req(self, url: str, headers: dict = None, param=None):
        if headers is None:
            headers = {}
        if "Authorization" not in headers:
            headers["Authorization"] = "Bearer " + self.tenant_access_token()
        return self._get_req(url, headers, param)

    def tenant_access_token(self, update=False):
        now_time = datetime.now()
        if update:
            self.storage["data"] = self._get_tenant_access_token()
            self.storage["time"] = now_time
        elif "data" not in self.storage or "time" not in self.storage:
            self.storage["data"] = self._get_tenant_access_token()
            self.storage["time"] = now_time
        elif (now_time - self.storage["time"]).seconds >= 1800:
            self.storage["data"] = self._get_tenant_access_token()
            self.storage["time"] = now_time
        return self.storage["data"]


    # 操作飞书多维表格
    def bitable_create(self, app_token: str, table_id: str, records: list):
        """用于在多维表格上批量新建数据，该方法的参数和 records 的组织方式请参考飞书的 API 文档"""

        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_create"
        # API: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/bitable-v1/app-table-record/batch_create

        rsp_dict = self.post_req(
            url, req_body={
                "records": records
            }
        )

        if rsp_dict.get("code", -1) != 0:
            print("Error[bitable_batch_create]", rsp_dict.get("code"), rsp_dict.get("msg"))
            return []
        return rsp_dict.get("data").get("records")

    def bitable_create_all(self, app_token: str, table_id: str, records: list):
        """用于在多维表格上新建一大批的数据"""
        for k in range(0, len(records), MAX_OPS):
            self.bitable_create(app_token, table_id, records=records[k:k + MAX_OPS])

    def bitable_update(self, app_token: str, table_id: str, records: list):
        """用于在多维表格上批量更新数据，该方法的参数和 records 的组织方式请参考飞书的 API 文档"""

        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_update"
        # API: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/bitable-v1/app-table-record/batch_update

        rsp_dict = self.post_req(
            url, req_body={
                "records": records
            }
        )

        if rsp_dict.get("code", -1) == 0:
            return rsp_dict.get("data").get("records")
        else:
            print("Error[bitable_batch_update]", rsp_dict["msg"])
            return []

    def bitable_update_all(self, app_token: str, table_id: str, records: list):
        r"""
        用于在多维表格上更新一大批的数据
        records: {"records_id": records_id, "fields": {} }
        """
        for k in range(0, len(records), MAX_OPS):
            self.bitable_update(app_token, table_id, records=records[k:k + MAX_OPS])

    def bitable_list(self, app_token: str, table_id: str, filter_dict: dict = None, page_token=""):
        """用于在多维表格上获取记录，该方法的参数请参考飞书的 API 文档"""
        if filter_dict is None:
            filter_dict = {}

        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records"
        # API: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/bitable-v1/app-table-record/list
        param = {
            "page_size": MAX_OPS,
            **filter_dict
        }
        if page_token != "":
            param["page_token"] = page_token
        rsp_dict = self.get_req(url, param=param)

        if rsp_dict.get("code", -1) == 0:
            data = rsp_dict.get("data")
            page_token = data.get("page_token", "") if data.get("has_more", False) else False
            return data.get("items"), page_token
        else:
            print("Error[get_records]", rsp_dict["msg"])
            return [], False

    def bitable_list_all(self, app_token: str, table_id: str, filter_dict: dict = None):
        """用于在多维表格上获取所有的记录"""
        page_token = ""
        ret_records = []
        while 1:
            records, page_token = self.bitable_list(app_token, table_id, filter_dict, page_token)
            if records is not None:
                ret_records.extend(records)
            if not page_token:
                break
        return ret_records

    def bitable_delete(self, app_token: str, table_id: str, records: list) -> int:
        """用于在多维表格上批量删除，该方法的参数请参考飞书的 API 文档"""

        if records is None:
            return -1
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/records/batch_delete"
        # API: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/bitable-v1/app-table-record/batch_delete

        rsp_dict = self.post_req(
            url, req_body={
                "records": [each["id"] for each in records]
            }
        )

        if rsp_dict.get("code", -1) == 0:
            _ = rsp_dict.get("data")
            return 0
        else:
            print("Error[del_items]", rsp_dict["msg"])
            return rsp_dict.get("code", -1)

    def bitable_delete_all(self, app_token: str, table_id: str, filter_dict: dict = None) -> None:
        """删除某个多维表格中的所有符合条件的数据

        TODO: 可能需要先 list 再 delete 来提升性能

        """

        if filter_dict is None:
            filter_dict = {}
        records, has_more = self.bitable_list(app_token, table_id, filter_dict)
        time.sleep(WAITING_TIME)
        self.bitable_delete(app_token, table_id, records)
        time.sleep(WAITING_TIME)
        if has_more:
            self.bitable_delete_all(app_token, table_id, filter_dict)

    def bitable_field_delete_all(self, app_token: str, table_id: str) -> None:
        """用于在多维表格上删除所有字段，该方法的参数请参考飞书的 API 文档"""

        items = self.bitable_field_list_all(app_token, table_id)
        for field in items:
            self.bitable_field_delete(app_token, table_id, field["field_id"])

    def bitable_field_delete(self, app_token: str, table_id: str, field_id: str) -> None:
        """用于在多维表格上删除字段，该方法的参数请参考飞书的 API 文档"""

        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/fields/{field_id}"
        # API: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/bitable-v1/app-table-field/delete

        rsp_dict = self.delete_req(url)
        if rsp_dict.get("code", -1) != 0:
            print("Error[bitable_delete_field]", rsp_dict.get("code"), rsp_dict.get("msg"))

    def bitable_field_list_all(self, app_token: str, table_id: str) -> list:
        """用于在多维表格上列出所有字段，该方法的参数请参考飞书的 API 文档"""

        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/fields"
        # API: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/bitable-v1/app-table-field/list
        rsp_dict = self.get_req(url)
        if rsp_dict.get("code", -1) != 0:
            print("Error[bitable_list_all_fields]", rsp_dict.get("code"), rsp_dict.get("msg"))
        return rsp_dict.get("data", {"items": []}).get("items")

    def bitable_field_create(
        self, app_token: str, table_id: str, field_name: str, field_type: int = 1,
        field_property: dict = None
    ) -> None:
        """用于在多维表格上增加字段，该方法的参数请参考飞书的 API 文档"""
        if field_name == "时间":
            field_type = 1001
            field_property = {
                "date_formatter": "yyyy-MM-dd HH:mm"
            }
        url = f"https://open.feishu.cn/open-apis/bitable/v1/apps/{app_token}/tables/{table_id}/fields"
        # API: https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/bitable-v1/app-table-field/create
        if field_property is not None:
            if "formatter" in field_property:
                formatter = str(field_property["formatter"])
                del field_property["formatter"]
                rsp_dict = self.post_req(
                    url, req_body={
                        "field_name": field_name,
                        "type": field_type,
                        "formatter": formatter,
                        "property": field_property
                    }
                )
            else:
                rsp_dict = self.post_req(
                    url, req_body={
                        "field_name": field_name,
                        "type": field_type,
                        "property": field_property
                    }
                )
        else:
            rsp_dict = self.post_req(
                url, req_body={
                    "field_name": field_name,
                    "type": field_type,
                }
            )
        if rsp_dict.get("code", -1) != 0:
            if rsp_dict.get("code") != 1254014:
                print("Error[bitable_create_field]", rsp_dict.get("code"), rsp_dict.get("msg"))

    def add_bitable(self, table_name: str, link: str, comment: str = "") -> None:
        """用于存储多维表格的配置信息

        :param table_name: 开发者定义的多维表格名
        :param link: 浏览器中多维表格的地址
        :param comment: 额外的备注信息
        :return:
        """
        if table_name in self._bitable_dict:
            print("Error! Table name {} has been saved in config.".format(table_name))
            return
        link_end = link.split("/")[-1]
        app_token = link_end.split("?")[0]
        params = link_end.split("?")[-1].split('&')
        table_id = ""
        for param in params:
            try:
                if param.split("=")[0] == 'table':
                    table_id = param.split("=")[1]
            except IndexError:
                pass
        if table_id == "":
            print("Error! Table id is not been found")
            return
        self._bitable_dict[table_name] = {
            "app_token": app_token,
            "table_id": table_id,
            "comment": comment
        }

    def bitable(self, table_name: str="default"):
        """

        :param table_name: 开发者定义的多维表格名
        :return: app_token, table_id
        """
        if table_name not in self._bitable_dict:
            raise KeyError("未找到名为{}的多维表格".format(table_name))
        item = self._bitable_dict[table_name]
        return item["app_token"], item["table_id"]

    def send(self, res: dict | list, table_name: str="default"):
        if self.show_info:
            print("*" * 80)
        print("正在使用飞书 API 发送数据, 下面会进行若干 Post 请求, 请保持网络通畅")
        
        app_token, table_id = self.bitable(table_name)
        fields = set([f["field_name"] for f in self.bitable_field_list_all(app_token, table_id)])
        
        time_now = datetime.now().timestamp()
        if not isinstance(res, list):
            res = [res]
        records_to_create = []
        for res_item in res:
            if "time" not in res_item and "时间" not in res_item:
                res_item = {
                    **res_item,
                    "时间": round(time_now * 1000)
                }

            for key, value in list(res_item.items()):
                if inspect.isfunction(value):
                    res_item[key] = str(value.__name__)
                elif isinstance(value, (int, float, bool)):
                    res_item[key] = value
                else:
                    res_item[key] = str(value)
                
                if key not in fields:
                    if isinstance(value, (int, float)):
                        field_type = 2
                    else:
                        field_type = 1
                    self.bitable_field_create(app_token, table_id, key, field_type)
                    fields.add(key)
            
            records_to_create.append({"fields": res_item})
        
        print(f"正在批量添加 {len(records_to_create)} 条记录...")
        self.bitable_create_all(app_token, table_id, records_to_create)

        time.sleep(WAITING_TIME)
        if self.show_info:
            print("*" * 80)
            print()


lark_config_kgextraction = dict(

)

lark_config_kgtooluse = dict(

)


class EmbModelHandlerV2:
    def __init__(self, model_name):
        self.model = self.load_model(model_name)

    def load_model(self, model_name):
        print(f"Loading model: {model_name}")
        model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        print(f"Model {model_name} loaded.")
        return model.to('cuda')

    def get_batch_embeddings(self, texts, batch_size=32):
        all_embeddings = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Processing Batches"):
            batch_texts = texts[i:i + batch_size]
            embeddings = self.model.encode(batch_texts, batch_size=batch_size, convert_to_tensor=True, normalize_embeddings=True)
            all_embeddings.append(embeddings.cpu().numpy())
        return np.vstack(all_embeddings)
    
    def unload_model(self):
        """
        释放模型占用的GPU显存。
        """
        if self.model:
            print("Unloading model and freeing GPU memory...")
            # 将模型移动到CPU（可选）
            self.model.to('cpu')
            # 删除模型对象
            del self.model
            # 清空PyTorch的缓存显存
            torch.cuda.empty_cache()
            # 进行垃圾回收，释放Python中的未引用对象
            gc.collect()
            print("Model has been unloaded and GPU memory has been freed.")
        else:
            print("No model to unload.")
class KnowledgeGraph():

    def __init__(self):
        self.startentitydict=defaultdict(list)
        self.relationdict=defaultdict(list)
        self.enddict=defaultdict(list)
        self.startrelationdict=defaultdict(list)
        self.nodetypedict=defaultdict(set)
        self.relationtypedict=defaultdict(set)
    def add(self,newkg):
        ###combine two kgs
        
        self.add_links(newkg.getall())
    def add_relations(self,relations,type):
        for relation in relations:
            self.relationtypedict[type].add(relation)

    def add_link(self,triple,starttype=None,endtype=None,relationtype=None):
        h,r,t=triple
        alllinks=self.getall()
        alllinksstr=["<>".join(link) for link in alllinks]
        newtriplestr="<>".join(triple)
        if newtriplestr not in alllinksstr:
            self.startentitydict[h].append([r,t])
            self.relationdict[r].append([h,t])
            self.enddict[t].append([h,r])
            self.startrelationdict[f"{h}-{r}"].append(t)
            if starttype is not None:
                self.nodetypedict[starttype].add(h)
            if endtype is not None:
                self.nodetypedict[endtype].add(t)
            if relationtype is not None:
                self.relationtypedict[relationtype].add(r)
    def add_links(self,triples,starttype=None,endtype=None,relationtype=None):
        for triple in triples:
            self.add_link(triple,starttype,endtype,relationtype)
    def get_links(self,input,by="start"):
        output=[]
        if by=="start":
            rts=self.startentitydict[input]
            for r,t in rts:
                output.append([input,r,t])
        elif by=="relation":
            hts=self.relationdict[input]
            for h,t in hts:
                output.append([h,input,t])
        elif by=="end":
            hrs=self.enddict[input]
            for h,r in hrs:
                output.append([h,r,input])

        return output
    def get_by_start_relation(self,start,relation):
        endentitys=self.startrelationdict[f"{start}-{relation}"]
        output=[]
        for endentity in endentitys:
            output.append([start,relation,endentity])
        return output
    def check_start_entity(self,start):
        return start in self.startentitydict.keys()
    def savetodisk(self,path):
        with open(path,"w") as f:
            for h,rts in self.startentitydict.items():
                for r,t in rts:
                    triple=[h,r,t]
                    f.write(str(triple))
                    f.write("\n")
        
    def loadfromdisk(self,path):
        ####not completed ,have no node type
        with open(path,"r") as f:
            for line in f:
                triple=eval(line)
                self.add_link(triple)
                    
    
    def getall(self):
        output=[]
        for h,rts in self.startentitydict.items():
            for r,t in rts:
                output.append([h,r,t])
        return output
    def get_relations(self):
        relations=set()
        for link in self.getall():
            relations.add(link[1])

        return relations

    def __repr__(self):
        return self.getall()
def has_unpaired_tag(text, open_pattern, close_pattern):
    open_tags = re.findall(open_pattern, text)
    close_tags = re.findall(close_pattern, text)
    return len(open_tags) > len(close_tags)

def load_model(model_name):
    print(f"Loading model: {model_name}")
    if model_name == 'Qwen/Qwen2.5-7B-Instruct':
        model = LLM(model="Qwen/Qwen2.5-7B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-7B-Instruct')
        return model, tokenizer
    elif model_name == 'QwQ-32B':
        model = LLM(model="QwQ-32B",max_model_len=4096)
        tokenizer = AutoTokenizer.from_pretrained('QwQ-32B')
        return model, tokenizer
    elif model_name == 'Qwen/Qwen2.5-32B-Instruct':
        model = LLM(model="Qwen/Qwen2.5-32B-Instruct",max_model_len=4096)
        tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-32B-Instruct')
        return model, tokenizer
    elif model_name=="Llama-3.1-8B":
        model = LLM(model="Meta-Llama-3.1-8B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained('Meta-Llama-3.1-8B-Instruct')
        return model, tokenizer
    elif model_name == 'Qwen3-8B':
        model = LLM(model="Qwen3-8B")
        tokenizer = AutoTokenizer.from_pretrained('Qwen3-8B')
        return model, tokenizer
    elif model_name == 'Qwen3-32B':
        model = LLM(model="Qwen3-32B")
        tokenizer = AutoTokenizer.from_pretrained('Qwen3-32B')
        return model, tokenizer
    elif model_name == 'gpt-4o':
        client = OpenAI(api_key="", base_url="")
        return client, None
    elif model_name == 'o3-mini':
        client = OpenAI(api_key="", base_url="")
        return client, None
    elif model_name == 'deepseek-chat':
        client = OpenAI(api_key="", base_url="")
        return client, None
    elif model_name == 'deepseek-reasoner':
        client = OpenAI(api_key="", base_url="")
        return client, None
    print(f"Model {model_name} loaded.")



def extract_text(text, pattern, index=None):
   """
   从文本中提取匹配正则表达式的内容
   
   参数:
       text (str): 要搜索的文本
       pattern (str): 正则表达式模式，应包含一个捕获组
       index (int, optional): 需要返回的匹配结果的索引。如果为None，返回所有匹配结果
       default (any, optional): 当index指定但无法找到对应索引的结果时返回的默认值
   
   返回:
       如果index为None，返回所有匹配结果的列表；否则返回指定索引的匹配结果或默认值
   """
   matches = [match.group(1) for match in re.finditer(pattern, text, re.DOTALL)]
   
   if index is None:
       return matches
   
   if 0 <= index < len(matches):
       return matches[index]
   
   return text
    


def read_qas(file_path):
    """
    从文件中读取 QA 数据
    :param file_path: JSON 文件路径
    :return: QA 数据列表
    """

    with open(file_path, 'r', encoding='utf-8') as f:
        qas = json.load(f)
    return qas


def save_json_file(data, output_file):
    """
    将数据保存为 JSON 文件
    :param data: 要保存的数据
    :param output_file: 输出文件路径
    """
    dir_name = os.path.dirname(output_file)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)



def vllm_release_memory(model):
    from vllm.distributed.parallel_state import destroy_model_parallel, destroy_distributed_environment
    import ray
    import contextlib
    del model.llm_engine.model_executor
    del model
    destroy_model_parallel()
    destroy_distributed_environment()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    print("Successfully delete the llm pipeline and free the GPU memory!")


def search_similar_qa(embedding_test_q, index, db_qas, k=3):
    if index.ntotal == 0:
        return []

    unique_qas = []  # 存储最终的 (Q, similarity) 结果
    seen_qs = set()  # 记录已加入的 Q，避免重复
    offset = 0  # 迭代扩展搜索范围
    step = k  # 每次扩展搜索的步长
    total = index.ntotal  # 索引中的总向量数

    while len(unique_qas) < k:
        current_k = min(k + offset, total)  # 确保不会超出索引范围
        if offset >= total:
            break  # 防止死循环

        distances, indices = index.search(embedding_test_q, current_k)  # Faiss 搜索
        results = [
            (db_qas[idx], distances[0][i])  # 获取问题 Q 和相似度
            for i, idx in enumerate(indices[0])
            if idx >= 0 and idx < len(db_qas)  # 过滤掉无效索引 -1
        ]

        for qa, similarity in results:
            if qa['Q'] not in seen_qs:  # 避免重复 Q
                seen_qs.add(qa['Q'])
                unique_qa = {
                    "Q": qa["Q"],
                    "A": qa["A"],
                    "similarity": float(similarity)
                }
                if "names" in qa:
                    unique_qa["names"] = qa['names']
                unique_qas.append(unique_qa)

            if len(unique_qas) >= k:  # 找到足够的不同 Q
                return unique_qas

        offset += step  # 扩展搜索范围

    return unique_qas  # 返回最终结果


def create_faiss_index(embeddings):
    """
    创建基于余弦相似度的 Faiss 索引
    :param embeddings: 向量化后的数据
    :return: Faiss 索引对象
    """
    print(f"Creating Faiss index with {embeddings.shape[0]} embeddings...")
    dimension = embeddings.shape[1]
    # 1. 创建使用内积(dot product)的索引
    index = faiss.IndexFlatIP(dimension)
    # 2. 对输入向量进行 L2 归一化，这样内积就等价于余弦相似度
    faiss.normalize_L2(embeddings)
    # 3. 添加归一化后的向量到索引中
    index.add(embeddings)
    print("Faiss cosine similarity index created.")
    return index


def print_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            total_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3  # GB
            allocated_memory = torch.cuda.memory_allocated(i) / 1024**3  # GB
            cached_memory = torch.cuda.memory_reserved(i) / 1024**3  # GB
            
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Total memory: {total_memory:.2f} GB")
            print(f"  Allocated memory: {allocated_memory:.2f} GB")
            print(f"  Cached memory: {cached_memory:.2f} GB")
            print(f"  Free memory: {total_memory - allocated_memory:.2f} GB")
    else:
        print("CUDA is not available")
