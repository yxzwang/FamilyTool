# FamilyTool
FamilyTool: A Multi-hop Personalized Tool Use Benchmark.

Pipeline code comming soon.
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