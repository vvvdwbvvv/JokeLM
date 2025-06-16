import ast
import json

input_path = "data/message.txt"
output_path = "data/message.jsonl"

with open(input_path, "r", encoding="utf-8") as f:
    data_list = ast.literal_eval(f.read())

with open(output_path, "w", encoding="utf-8") as f:
    for item in data_list:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
