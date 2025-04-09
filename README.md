# FamilyTool
FamilyTool: A Multi-hop Personalized Tool Use Benchmark
## Data Path
### Query-Answer Pair
For dataset name as $DATASET, data with golden KG is put in '```datasets/$DATASET/data_goldenKG.jsonl```. Data format for each line is 
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
### KG
KG is put in ```KG/familykg.txt```. Each line is a link in KG.