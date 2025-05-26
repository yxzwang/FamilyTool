from utils import KnowledgeGraph
from collections import defaultdict
import re
import json
import argparse
from utils import extract_links_func 
from utils import save_json_file,read_json
from utils import lark_config_kgextraction,LarkReport




###get postprocess 
relation_mappingdict={}




def cleanrelation(outputpath):
    outputpath= re.sub(r'<.*?>', '', outputpath)
    for key,value in relation_mappingdict.items():
        outputpath=outputpath.replace(key,value)
    return outputpath
def extractpath(start,outputpath,allkg):##outputpath is [a,b]
    ###遇到错误需要用retrieval得加上

    extract_links=[]
    extract_sign=""
    ####extract_links=[[h,r,q]]然后不停在后面加入extract_links=[[h,r,q],[h,r,q]]。如果是多个起点就[[h,r,q],[h,r,q]]开始。
    for i,relation in enumerate(outputpath):
        if i==0:
            if not allkg.check_start_entity(start):
                return extract_links,extract_sign+"<bad_extract>"
            else:
                candidates=allkg.get_links(start,by="start")

            if f"{start}-{relation}" not in allkg.startrelationdict.keys() or len(allkg.get_by_start_relation(start,relation))<1:
                extract_links=[[candidate]for candidate in candidates]
                nextstarts=[candidate[-1] for candidate in candidates]
                extract_sign+="<bad_extract>"
            else:
                
                extract_links=[[allkg.get_by_start_relation(start,relation)[0]]]
                nextstarts=[allkg.get_by_start_relation(start,relation)[0][-1]]
                
        else:
      
            if len(nextstarts)<1:
                extract_sign+="<bad_extract>"
                continue
            for j,nextstart in enumerate(nextstarts):
                if not allkg.check_start_entity(nextstart):
                    extract_sign+="<bad_extract>"
                    continue
                if f"{nextstart}-{relation}" not in allkg.startrelationdict.keys() or len(allkg.get_by_start_relation(nextstart,relation))<1:
                    if "<bad_extract>" not in extract_sign:
                        extract_links[j].extend([link for link in allkg.get_links(nextstart,by="start")])
                        extract_sign+="<bad_extract>"
                    else:
                        
                        continue
                else:
                    extract_links[j].append(allkg.get_by_start_relation(nextstart,relation)[0])
                    nextstarts[j]=allkg.get_by_start_relation(nextstart,relation)[0][-1]

    if "<bad_extract>" in extract_sign:
        extract_sign="<bad_extract>"
    return extract_links,extract_sign ####[[h,r,q],[h,r,q],[[h,r,q],[h,r,q]],[[h,r,q],[h,r,q]] 很多条path4

def torealname(path):
    # for key,value in entity2realdict.items():
    #     path=path.replace(key,value)
    return path

def cleankgsearch(kgsearch):
    cleaned_kgsearch=kgsearch.replace('"', '').replace("'", '').replace("`", '').replace("‘", '').replace("’", '').replace("“", '').replace("”", '').replace(" ","").replace("，",",").replace("】","]").replace("【","[").replace("，",",")
    
    return cleanrelation(cleaned_kgsearch)
def merge_with_previous(strings):
    ###将"全部"和前面的合并
    new_list = []  # 创建一个空列表用于存放最终结果
    i = 0  # 初始化索引
    while i < len(strings):  # 遍历输入列表
        if strings[i] == "全部":  # 判断当前元素是否为"全部"
            if i > 0:  # 确保不是第一个元素，因为第一个元素没有前面的元素可以合并
                # 将当前的"全部"与前面一个元素合并，并添加到新列表中
                new_list[-1] = new_list[-1] + strings[i]
            else:
                # 如果"全部"是第一个元素，直接添加到新列表
                new_list.append(strings[i])
        else:
            # 如果当前元素不是"全部"，直接添加到新列表
            new_list.append(strings[i])
        i += 1  # 索引加1，继续处理下一个元素
    return new_list


def examinekg_output(input_string):
    # 使用正则表达式匹配 target 和 start

    start_match = re.search(r'start=([^,]+)', input_string)
    
    # 提取 path 部分
    path_match = re.search(r'path=\[([^\]]+)\]', input_string)
    
    # 处理提取结果

    
    if start_match:
        start = start_match.group(1).strip()
    else:
        start = None
    
    if path_match:
        path = [item.strip() for item in path_match.group(1).split(',')]
            ##合并“全部”和前面的relation
        if len(path)<2:
            pass
        else:
            path=merge_with_previous(path)

    else:
        path = None
    return  start, path



def extractkgsearch(temp_response):
    pattern= r'KG.search\((.*?)\)'
    matches=re.findall(pattern,temp_response)
    return matches

# def post_process(extracted_paths):
#     ####extracted_paths=[[[h,r,q],[h,r,q]]]

#     outputs=[]

#     for extract_path in extracted_paths:
         
        
            
#             # cardevicesforlocation=set(allkg.startrelationdict[f"{finalentity}-附属对手件"])##附属对手件

#         output=f"{extract_path}"
#         replacerealname=torealname(output)
#         outputs.append(replacerealname)
#     outputstr=f"The extra information for the query is {len(outputs)} path(s).\n"
#     for i in range(len(outputs)):

    
#         formatoutput=f"The {i+1}th path is : ({outputs[i]})\n"
#         outputstr+=formatoutput
#     return outputstr
#######################################KG_search##################################
def filter_longest_lists(nested_list):
    """
    判断嵌套列表中的子列表是否长度相同，如果不同，返回最长的子列表
    :param nested_list: 嵌套列表
    :return: 如果所有子列表长度相同，返回原嵌套列表；否则返回最长的子列表组成的列表
    """
    # 获取所有子列表的长度
    lengths = [len(sublist) for sublist in nested_list]

    # 检查所有子列表长度是否相同
    if len(set(lengths)) == 1:
        return nested_list
    else:
        # 找到最长的长度
        max_length = max(lengths)
        # 筛选出最长的子列表
        longest_lists = [sublist for sublist in nested_list if len(sublist) == max_length]
        return longest_lists
def KG_search(temp_response,KGretrieval_type,kgpath,model_handler=None):
    familykg=KnowledgeGraph()
    familykg.loadfromdisk(kgpath)
    #######################################################
    allkg=KnowledgeGraph()
    allkg.add(familykg)
    
    matches=extractkgsearch(temp_response)
    if len(matches)<1:
        return ""
    outputpaths=[]
    for extracted_kg_response in matches:
        cleaned_response=cleankgsearch(extracted_kg_response)
        startentity_output,paths_output=examinekg_output(cleaned_response)

        if paths_output is not None:

        #### relation retrieval.
            if KGretrieval_type=="exact":
                extracted_paths,extract_sign=extractpath(startentity_output,paths_output,allkg)
            elif KGretrieval_type=="relation_retrieval":
                extracted_paths,extract_sign=extractpath_relationretrieval(startentity_output,paths_output,familykg,allkg,model_handler)

            if len(extracted_paths)>0:
                extracted_paths=filter_longest_lists(extracted_paths)
            for extract_path in extracted_paths:
                outputpaths.append(f"({extract_path})")
                outputpaths.append(extract_sign)

        else:
            outputpaths.append("")
            
            outputpaths.append("<bad_extract>")

    outputstr="".join(outputpaths)
    return outputstr
#######################################KG_search#################################    
#####For path retrieval#############################################################################################    
from tqdm import tqdm
import numpy as np
from utils import EmbModelHandlerV2, create_faiss_index

embedding_model_name="jinaai/jina-embeddings-v3"



import os



def search_similar_relations(embedding_test, index, embdb_qas, k=3,):
    """
    使用 Faiss 索引检索与查询最相似的 QA
    :param embedding_test: 测试嵌入向量
    :param index: Faiss 索引
    :param test_qas: 测试集 QA 列表
    :param k: 返回的相似结果数量
    :param dedup: 是否去重
    :return: 相似的 QA 列表
    """

    # 循环检索，直到获取到 k 个唯一的结果

    distances, indices = index.search(embedding_test, k )
    results = [embdb_qas[idx] for idx in indices[0]]
        
        

    return results[:k]
def generate_fakerelationdict(fake_relations,true_relations,model_handler,k=3):
    fakerelationdict={}
    db_q_embbedings = model_handler.get_batch_embeddings(true_relations)
    index = create_faiss_index(db_q_embbedings)
    for fakerelation in tqdm(fake_relations, desc="fake relation retrieving...", total=len(fake_relations)):
        test_q_embedding = model_handler.get_batch_embeddings([fakerelation])[0]
        test_q_embedding_np = np.array(test_q_embedding).reshape(1, -1)
        top_k_results = search_similar_relations(test_q_embedding_np, index, true_relations, k)
        fakerelationdict[fakerelation]=top_k_results
    # 释放显存
    
    print(fakerelationdict)

    return fakerelationdict
import itertools
def generate_true_paths(outputpath,fakerelationdict,true_relations):
    
    replaced_path=[]
    for i,relation in enumerate(outputpath):
        if relation in true_relations:
            replaced_path.append([relation])
        else:
            replaced_path.append(fakerelationdict[relation])
    extracted_true_relation_paths=[list(combination) for combination in itertools.product(*replaced_path)]
    return extracted_true_relation_paths


def extractpath_relationretrieval(start,outputpath,familykg,allkg,model_handler):
    ###get good relation paths
    

    all_generated_fake_relations=set()


    goodrelations=list(familykg.get_relations())
    k_kgtype=3
    for i,relation in enumerate(outputpath):
        if relation not in goodrelations:
            all_generated_fake_relations.add(relation)

    all_generated_fake_relations=list(all_generated_fake_relations)
    if len(all_generated_fake_relations)>0:
        fakerelation_dict=generate_fakerelationdict(all_generated_fake_relations,goodrelations,model_handler,k=k_kgtype)
        extracted_true_relation_paths=generate_true_paths(outputpath,fakerelation_dict,goodrelations)
        outputpath=extracted_true_relation_paths
    else:
        ###不过检索器，加速。
        outputpath=[outputpath]
        pass

    # model_handler.unload_model()
    #########extracted paths
    
    extracted_paths=[]
    extracted_relations=[]
    ####extract_links=[["h,r,q"],["h,r,q"],["h,r,q"]]然后不停在后面加入extract_links=[["h,r,q","h,r,q"],["h,r,q","h,r,q"],["h,r,q","h,r,q"]]
    for origpath in outputpath:
        path=merge_with_previous(origpath)
        newpath=[]
        newrelations=[]
        continuepath=0
        nextstart=start
        for i,relation in enumerate(path):


            if f"{start}-{relation}" not in allkg.startrelationdict.keys() or len(allkg.get_by_start_relation(nextstart,relation))<1:
                continue
            else:
                newrelations.append(relation)
                newpath.append(allkg.get_by_start_relation(nextstart,relation)[0])
                nextstart=allkg.get_by_start_relation(nextstart,relation)[0][-1]

        # print(path)
        if len(newrelations)>0:
            extracted_relations.append(newrelations)
            extracted_paths.append(newpath)

    drop_duplicatepaths=[] 
    endentitys=[]
    for path in extracted_paths:
        endentity=path[-1][-1]
        if endentity not in endentitys:
            endentitys.append(endentity)
            drop_duplicatepaths.append(path)
        
    # extracted_relations=list(dict.fromkeys(map(tuple, extracted_relations)))
    return drop_duplicatepaths,"" ####[["h,r,q","h,r,q"],["h,r,q","h,r,q"],["h,r,q","h,r,q"]] 很多条path4

def test():
    testkgsearch=[
        "KG.search(start=\"Bob\", path=[\"<relationship>mother\",\"<relationship>prefer_dinnertime\"])",
                  "KG.search(start=\"Jack\", path=[\"<relationship>son\",\"<relationship>prefer_dinnertime\"])",
                  "KG.search(start=\"Bob\", path=[\"grandfather\",\"<relationship>prefer_dinnertime\"])", 
                  "KG.search(start=\"Bob\", path=[\"<relationship>mother\",\"<relationship>prefer_dinnertime\"])KG.search(start=\"Jack\", path=[\"<relationship>son\",\"<relationship>prefer_dinnertime\"])",
                  
                  ]

    extract_results=[]
    for kgsearch in testkgsearch:
        testextract=KG_search(kgsearch,"exact")
        extract_results.append(testextract)

    for result in extract_results:
        print(result)
        print("\n")
def read_jsonl(file_path):
    """
    读取 JSONL 文件，每行是一个 JSON 对象，返回一个列表
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append(json.loads(line))  # 将每一行的 JSON 数据解析并添加到列表中
    return data
def save_jsonl(data, file_path):
    """
    将数据保存到 JSONL 文件
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for item in data:
                # 将每个字典转换为 JSON 字符串，并写入文件
                json.dump(item, f, ensure_ascii=False)
                f.write("\n")  # 每个 JSON 对象占一行
        print(f"数据已成功保存到 {file_path}")
    except Exception as e:
        print(f"保存 JSON 文件时发生错误: {e}")
def calculate_f1em_score_single_sample(y_true, y_pred):
    """
    计算单个样本的 F1 分数
    :param y_true: 真实标签，形状为 (n_labels,)
    :param y_pred: 预测标签，形状为 (n_labels,)
    :return: F1 分数
    # 示例输入
    y_true = [1, 0, 1, 0, 1]  # 真实标签
    y_pred = [1, 0, 0, 1, 1]  # 预测标签
    """
    # 确保输入是 NumPy 数组
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # 计算 TP, FP, FN
    tp = np.sum(y_true * y_pred)  # 真正例
    fp = np.sum((1 - y_true) * y_pred)  # 假正例
    fn = np.sum(y_true * (1 - y_pred))  # 假负例

    # 计算精确率和召回率
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # 计算 F1 分数
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    em=1 if sum(y_true == y_pred)==len(y_true) else 0
    return f1,em
def get_entitys(links):
    entitys=set()
    for link in links:
        h,r,t=link
        entitys.add(h)
        entitys.add(t)
    return list(entitys)


def changelink2label(goldlinks,extract_links,gold_response):
    goldlinkstr=set(["<>".join(link) for link in goldlinks])
    extract_linkstr=set(["<>".join(link) for link in extract_links])
    all_classes=list(set(goldlinkstr | extract_linkstr))
    y_true=np.zeros((len(all_classes),))
    y_pred=np.zeros((len(all_classes),))
    for link in goldlinkstr:
        y_true[all_classes.index(link)]=1
    for link in extract_linkstr:
        y_pred[all_classes.index(link)]=1

    ### check what entity is used in gold_response
    entitysingoldlink=get_entitys(goldlinks)
    goldentitys=[entity for entity in entitysingoldlink if entity in gold_response]

    extracted_entitys=get_entitys(extract_links)

    coverage_sample=1
    for goldentity in goldentitys:
        if goldentity not in extracted_entitys:
            coverage_sample=0
            break
    



    return y_true.astype(int),y_pred.astype(int),coverage_sample


def metric_KG_extraction(intermediate_file,golden_jsonl_data,KG_retrieval_type,kgpath):
    total=0

    no_hallucination=0
    coverage=0
    Exactmatch=0
    f1=0
    formaterror=0
    extracted_results=[]
    notcoveragesamples=[]
    if KG_retrieval_type=="exact":
        model_handler=None
    else:
        model_handler=EmbModelHandlerV2(embedding_model_name)
    for data,golden_data in zip(intermediate_file,golden_jsonl_data):

        extract_result=KG_search(data["response"],KG_retrieval_type,kgpath,model_handler)
        extracted_results.append(extract_result)
        total+=1
        goldresponse=str(golden_data[-1]["content"])
        goldquery=golden_data[2]["content"]
        kg_match = re.search(r'\(.*?\)', goldquery)
        goldKG = kg_match.group(0)
        if extract_result=="":
            formaterror+=1
        elif not "bad_extract" in extract_result:
            no_hallucination+=1
        goldlinks=extract_links_func(goldKG)
        extract_links=extract_links_func(extract_result)
        y_true,y_pred,coverage_sample=changelink2label(goldlinks,extract_links,goldresponse)
        f1_sample,em_sample=calculate_f1em_score_single_sample(y_true,y_pred)
        f1+=f1_sample
        Exactmatch+=em_sample
        if em_sample==0:
            import ipdb
            # ipdb.set_trace()
            notcoveragesamples.append({"id":golden_data[0]["content"],"query":goldquery,"response":data["response"],"KG":goldlinks})
        coverage+=coverage_sample
    output=[Exactmatch,f1,no_hallucination,coverage,formaterror]

    no_hallucination_total=total-formaterror
    outputratio=[Exactmatch/total,f1/total,no_hallucination/no_hallucination_total,coverage/total,formaterror/total]###只在检测到format的时候测试hallucination
    return (total,output, outputratio),extracted_results,notcoveragesamples

def KG_extraction_post(extraction_file_path,KG_retrieval_type,gold_jsonl_path,kgpath,args):
    lark=LarkReport(**lark_config_kgextraction)

    extraction_file_signal=extraction_file_path[:-5].split("/")[-1]
    intermediate_file=read_json(extraction_file_path)
    
    golden_jsonl_filepath=gold_jsonl_path
    golden_jsonl_signal=golden_jsonl_filepath[:-6].split("/")[-1]
    golden_jsonl_data=read_jsonl(golden_jsonl_filepath)

    ####get metric
    metricresult,extracted_results,notcoveragesamples=metric_KG_extraction(intermediate_file,golden_jsonl_data,KG_retrieval_type,kgpath)
    total,outputcounts,(Exactmatch,f1,no_hallucination,coverage,formaterror)=metricresult
    resultdict={

                "golden_jsonl":golden_jsonl_signal,
                "extraction_file":extraction_file_signal,
                "useprompt":args.useprompt,
                "model_name":args.model_name,
                "KG_retrieval_type":KG_retrieval_type,
                "em":Exactmatch,
                "f1":f1,
                "no_hallucination":no_hallucination,
                "coverage":coverage,
                "formaterror":formaterror,
                "counts":outputcounts,
                "total":total


    }

    save_json_file([resultdict],f"results/KG_extraction_results/reports/{KG_retrieval_type}/report_{golden_jsonl_signal}_{extraction_file_signal}_{KG_retrieval_type}_KGmetric.json")
    save_json_file(notcoveragesamples,f"results/KG_extraction_results/reports/notcoveragesamples/{KG_retrieval_type}/report_{golden_jsonl_signal}_{extraction_file_signal}_{KG_retrieval_type}_KGmetric.json")
    print("sending")
    lark.send(resultdict)
    print(f"KG extraction metrics saving to {f"results/KG_extraction_results/reports/{KG_retrieval_type}/report_{golden_jsonl_signal}_{extraction_file_signal}_{KG_retrieval_type}_KGmetric.json"}")
    print(metricresult)
    datawithextracted_KG=[]
    
    for extracted_KG,data in zip(extracted_results,golden_jsonl_data):
        querydict=data[2] ### role==user
        goldquery=querydict["content"]
        kg_match = re.search(r'\(.*?\)', goldquery)
        kg_info = kg_match.group(0)
        extracted_KG_query=goldquery.replace(kg_info,extracted_KG)
        querydict["content"]=extracted_KG_query
        data[2]=querydict
        datawithextracted_KG.append(data)

    if "hard" in golden_jsonl_signal:
        hardsignal="hard"
    else:
        hardsignal="easy"
    outputfile=f"datasets/intermediate_jsonls/{KG_retrieval_type}/{hardsignal}/{golden_jsonl_signal}_{extraction_file_signal}_{KG_retrieval_type}.jsonl"
    print(f"results saving to {outputfile} for tool-use generation evaluation")
    save_jsonl(datawithextracted_KG,outputfile)

def parse_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="执行 Two Model 流程")

 
    parser.add_argument('--model_name', type=str, default="Qwen2.5-7B-Instruct",  help="改写模型名称")  
    parser.add_argument('--data_path', type=str, default="datasets/familytool-b.jsonl",  help="goldjsonl文件path")
    parser.add_argument('--extracted_data_path', type=str, default="results/KG_extraction_results/base",  help="goldjsonl文件path")
    parser.add_argument('--kg_path', type=str, default="KGs/familykg-b.txt",  help="kg path")
    parser.add_argument('--k', type=int, default=3, help="top k")
    parser.add_argument('--KGretrieval_type', type=str, default="exact", help="",choices=["exact","relation_retrieval"])
    parser.add_argument('--use_template', type=bool,default=True, help="是否使用模板")  
    parser.add_argument('--no_think', action="store_true",default=False, help="for qwen3")  
    parser.add_argument('--useprompt', type=str, default="new_prompt", help="new prompt or old ones")
    args = parser.parse_args()

    return args
def main():
    args=parse_args()

    KG_retrieval_type=args.KGretrieval_type
    extraction_path=args.extracted_data_path
    if "hard" in extraction_path:
        args.data_path="datasets/familytool-e.jsonl"
        args.kg_path="KGs/familykg-e.txt"
    else:
        args.data_path="datasets/familytool-b.jsonl"
        args.kg_path="KGs/familykg-b.txt"
    gold_jsonl_path=args.data_path
    kgpath=args.kg_path
    if extraction_path.endswith(".json"):

        KG_extraction_post(extraction_path,KG_retrieval_type,gold_jsonl_path,kgpath,args)
    elif os.path.isdir(extraction_path):
        files = os.listdir(extraction_path)

        # 筛选出以.json结尾的文件，并拼接完整路径
        json_files = [os.path.join(extraction_path, file) for file in files if file.endswith('.json')]
        print(json_files)
        for json_file in json_files:
            print(f"extraction for file: {json_file}")
            KG_extraction_post(json_file,KG_retrieval_type,gold_jsonl_path,kgpath,args)
    else:
        print("not implemented")
    
if __name__=="__main__":
    main()
    

