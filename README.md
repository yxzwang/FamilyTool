# Updates
## 2025/04/26
Pipeline code is added. 
# FamilyTool
[FamilyTool: A Multi-hop Personalized Tool Use Benchmark](https://arxiv.org/abs/2504.06766). This work is one extended benchmark of [UnifiedToolHub](https://github.com/OpenMOSS/UnifiedToolHub).

## Data Path
### Query-Answer Pair
For dataset name as $DATASET, data with golden subKG is put in '```datasets/$DATASET/data_goldenKG.jsonl```. Data format for each line is 
```
[
        {"role": "id", "content": "$id"},
        {"role": "candidate_tools", "content": #candidate_tools},
        {"role": "user", "content":f "{query}, The extra information for the query is ({golden_KG})"},
        {"role": "tool_call", "content": [
            $golden_answer_dict
        ]}
    ]

```

The golden subKG used for each sample can be extracted by regular expression.
### KG
KG is put in ```KG/familykg.txt```. Each line is a triple in KG.

## KGETool Pipeline
We show the code for pipeline here, including 3 python files in ```pipeline/```. To get results you should run them in order. We use MTU-Bench for example here.
### KG extraction

```python pipeline/KG_extraction.py --data_path datasets/MTU-Bench/data_goldenKG.jsonl --model_name Qwen/Qwen2.5-7B-Instruct```


args:
```
'--model_name', support [Qwen/Qwen2.5-7B-Instruct,QwQ-32B,Llama-3.1-8B,gpt-4o,o3-mini,deepseek-chat,deepseek-reasoner].  For gpt and deepseek type model, you should set api_key and base_url in utils.py load_model function.
'--data_path', input data path.

```

Output file for next step is in ```results/KG_extraction_results/```
### Computing Extraction Metric and Generating intermediate file for KG-augmented Tool use 
 ```python pipeline/KG_extraction_post_process_make_intermediate_json.py --data_path results/KG_extraction_results/xxxxx.json --KGretrieval_type exact --k 3 ```

args:
```
'--data_path', output file of KG extraction step.
'--KGretrieval_type', choose from ["exact","relation_retrieval"]. Exact is for greedy search and relation_retrieval is for Relation Retrieval in the paper. Default is "exact".
'--k', top k for relation retrieval
```

Output jsonl file for next step is in ```datasets/MTU-Bench/intermediate_jsonls/{KG_retrieval_type}/```

### KG-augmented Tool use
```python pipeline/Tooluse_generation.py``` 

We use [UnifiedToolHub](https://github.com/OpenMOSS/UnifiedToolHub) Code for this step and prepare a clone in our repo in ```resources/UnifiedToolHub```.
test config is in ```resources/UnifiedToolHub/test.py```.Test LLM config is in ```test_models```. We have prepared some samples. For gpt and deepseek type model, you should set api_key and base_url 

 Add the jsonl file path from last step in the ```test_datasets```. If you only want to test results with golden KG, you can just use the initial jsonl file (```datasets/MTU-Bench/data_goldenKG.jsonl```) which is used as input of the pipeline and only run this step (see ```resources/UnifiedToolHub/test.py``` for the example).
