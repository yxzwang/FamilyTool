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
import torch
from tqdm import tqdm
from transformers import AutoModel



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
    elif model_name=="Llama-3.1-8B":
        model = LLM(model="Meta-Llama-3.1-8B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained('Meta-Llama-3.1-8B-Instruct')
        return model, tokenizer
    elif model_name == 'Qwen3-8B':
        model = LLM(model="Qwen3-8B")
        tokenizer = AutoTokenizer.from_pretrained('Qwen3-8B')
        return model, tokenizer

    elif model_name =="Qwen3-4B":
        model = LLM(model="Qwen3-4B")
        tokenizer = AutoTokenizer.from_pretrained('Qwen3-4B')
    
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
