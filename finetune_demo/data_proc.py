#%%
# 打开JSON Lines文件读取数据
import json

with open("../auto_data_process/training_set.jsonl", 'r', encoding='utf-8') as file, \
     open("./data/train.jsonl", 'w', encoding='utf-8') as train_file,\
     open("./data/dev.jsonl", 'r', encoding='utf-8') as dev_file:
    for line in file:
        # 解析每一行JSON数据
        record = json.loads(line)
        
        # 假设我们要提取的文本字段名为 'text'
        # 请根据您的JSON结构更改字段名
        instruction = record.get('instruction', '')
        input = record.get('input', '').rstrip('。')
        output = record.get('output', '')

        messages = {"messages": [{"role": "user", "content": f"{instruction}请编写主题为：{input}的微小说故事。"}, {"role": "assistant", "content": f"{output}"}]}
        train_file.write(json.dumps(messages, ensure_ascii=False) + '\n')
        
    # dev_messages = {"messages": [{"role": "user", "content": f"{instruction}请编写主题为：{input}的微小说故事。"}, {"role": "assistant", "content": f"{output}"}]}
    # dev_file.write(json.dumps(dev_messages, ensure_ascii=False) + '\n')
     
# %%

