
stories = [ 
"现代都市与古代时空相交，一个现代女画家在画一幅古代风景画时，意外穿越到了画中所描绘的朝代，她不仅卷入了宫廷的争斗中，还发现了这幅画背后的惊人秘密。",
"神秘岛屿生存冒险，一群来自现代社会的幸存者在飞机失事后流落到一个未知岛屿，他们必须利用各自的知识和技能生存下去，同时解开岛上隐藏的古老秘密。",
"现代奇幻背景，一个平凡的图书管理员偶然发现一本能预知未来的书，她如何利用这本书避免一场重大灾难，并揭示这本书的神秘来源与其背后的历史。",
"现代都市背景，一个被束缚在重复、无趣生活中的普通白领，逐渐发现身边的一系列巧合和神秘事件与他自己创作的小说有着千丝万缕的联系，最终揭露出前世今生不可思议的秘密。",
"古代背景，一个天资聪慧但性格古怪的谋士，通过看似不可能完成的任务逐渐赢得君主信任，在朝堂内外斗智斗勇，最终揭开宫廷深处的惊天大秘密的故事。",
"未来都市背景，一群热爱科幻和玄幻的青少年，通过一次虚拟世界的探险，逐渐发现现实中的城市竟然隐藏着一个被遗忘的外星文明，他们如何与这个文明沟通并挽救危机的故事。",
"中国古代背景，一个曾经的宫廷侍卫，被自己的朋友陷害而入狱，被指派一桩决定他生死的护卫任务，在过程中保护自己、寻找真相洗脱冤屈的故事。",
"唐朝背景，一位身穿不凡的刺客与被贬的公主携手合作，通过冒险和智慧，揭开朝廷内外重重阴谋，最终扭转乾坤并相伴一生的爱情故事。",
"古代背景，一位边境小村的普通少年，在偶然目睹一场刺杀后，被卷入朝廷与江湖势力的斗争中，通过不断成长和历练，揭开自身惊人身世并扭转战局的惊险旅程。",
"现代背景，一名退休侦探无法忍受平静的生活，他在一次老友聚会上接手了一桩谜团重重的老案件，运用他的经验和现代科技手段，最终破解了长达数十年的谜案，为受害者家属带来慰藉。"
  ]


tasks_prompt = [ 
"现代都市与古代时空相交，一个现代女画家在画一幅古代风景画时，意外穿越到了画中所描绘的朝代，她不仅卷入了宫廷的争斗中，还发现了这幅画背后的惊人秘密。",
"神秘岛屿生存冒险，一群来自现代社会的幸存者在飞机失事后流落到一个未知岛屿，他们必须利用各自的知识和技能生存下去，同时解开岛上隐藏的古老秘密。",
"现代奇幻背景，一个平凡的图书管理员偶然发现一本能预知未来的书，她如何利用这本书避免一场重大灾难，并揭示这本书的神秘来源与其背后的历史。",
"现代都市背景，一个被束缚在重复、无趣生活中的普通白领，逐渐发现身边的一系列巧合和神秘事件与他自己创作的小说有着千丝万缕的联系，最终揭露出前世今生不可思议的秘密。",
"古代背景，一个天资聪慧但性格古怪的谋士，通过看似不可能完成的任务逐渐赢得君主信任，在朝堂内外斗智斗勇，最终揭开宫廷深处的惊天大秘密的故事。",
"未来都市背景，一群热爱科幻和玄幻的青少年，通过一次虚拟世界的探险，逐渐发现现实中的城市竟然隐藏着一个被遗忘的外星文明，他们如何与这个文明沟通并挽救危机的故事。",
"中国古代背景，一个曾经的宫廷侍卫，被自己的朋友陷害而入狱，被指派一桩决定他生死的护卫任务，在过程中保护自己、寻找真相洗脱冤屈的故事。",
"唐朝背景，一位身穿不凡的刺客与被贬的公主携手合作，通过冒险和智慧，揭开朝廷内外重重阴谋，最终扭转乾坤并相伴一生的爱情故事。",
"古代背景，一位边境小村的普通少年，在偶然目睹一场刺杀后，被卷入朝廷与江湖势力的斗争中，通过不断成长和历练，揭开自身惊人身世并扭转战局的惊险旅程。",
"现代背景，一名退休侦探无法忍受平静的生活，他在一次老友聚会上接手了一桩谜团重重的老案件，运用他的经验和现代科技手段，最终破解了长达数十年的谜案，为受害者家属带来慰藉。"
  ]

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, TaskType, get_peft_model

import random

baijiaxing_top50 = [
    "赵", "钱", "孙", "李", "周", "吴", "郑", "王",
    "冯", "陈", "褚", "卫", "蒋", "沈", "韩", "杨",
    "朱", "秦", "尤", "许", "何", "吕", "施", "张",
    "孔", "曹", "严", "华", "金", "魏", "陶", "姜",
    "戚", "谢", "邹", "喻", "柏", "水", "窦", "章",
    "云", "苏", "潘", "葛", "奚", "范", "彭", "郎",
    "鲁", "韦"
]

def generate_prompt(idx):
    # 设置随机种子以确保每次生成的数字一致
    random.seed(idx)

    # 生成不同的随机数，用于问题四中的数字
    major_characters = random.choice([2, 3, 4])
    minor_characters = random.choice([3, 4, 5])
    plot_twists = random.choice([3, 4, 5])
    family_name = baijiaxing_top50[idx]

    prompt = (
        f"假如你是一位才华横溢的畅销小说作家，以独特的创作风格著称：现在要求你在给定主题下进行小说写作，字数在800字左右(字数不包括所有标点符号和换行符)。\n"
        f"要求：\n"
        f"1. 内容贴合主题。\n"
        f"2. 文本重复度低，不要出现大段无意义的重复，不要出现空格和换行符。\n"
        f"3. 语言通顺流畅，语句完整，运用华丽且流畅的语言描绘场景和人物。注意行文连贯，不要按点总结。\n"
        f"4. 故事情节曲折复杂：你需要精于编织情节，使故事层次丰富、悬念迭起，要有反转的事件发生。需要有{plot_twists}次情节反转、悬念以及人物关系发生{plot_twists}次颠覆性变化的事件发生。故事中需要出现至少{major_characters}个主要人物，故事中要有一个人姓{family_name}。\n"
        f"5. 故事内容要具有创新性，新颖不老套，充分发挥你的想象力。\n"
        f"6. 文章结构和情节完整，要包含事件的起因、经过和结尾，尤其是结尾部分，不能出现结尾中断。\n"
        f"7. 严格按照字数要求创作，字数在800字左右(字数不包括所有标点符号和换行符)，注意不要低于700字，也不要超过900字。\n"
        f"下面请根据主题，开始进行小说创作。"
    )
    print("prompt:", prompt)

    return prompt


from openai import OpenAI
client = OpenAI(
    base_url = 'http://0.0.0.0:8000/v1',
    api_key="sk-xxx"
)

import re

def sentence_segmentation(text):
    # 定义句子分隔符的正则表达式
    sentence_delimiters = r'[, ，。！？]+'
    
    # 使用re.split进行句子分割
    # 忽略空字符串，使用filter过滤掉空字符串
    sentences = list(filter(None, re.split(sentence_delimiters, text)))
    
    return sentences

from collections import Counter
def is_sentence_repetition(text, threshold=7):
    """
    判断文本中是否存在大量重复的句子。
    
    :param text: str, 输入的文本。
    :param threshold: float, 重复句子的最小比例阈值。
    :return: bool, 如果重复句子的比例超过阈值，则返回True。
    """
    if len(text) ==0:
        return True
    # 使用正则表达式去除文本中的多余空格和标点符号
    sentences = sentence_segmentation(text)
    # 计数句子出现的频率
    sentence_counts = Counter(sentences)
    if max(sentence_counts.values()) > threshold:
        print(sentence_counts)
    # 如果重复句子的比例超过阈值，则认为存在大量重复
    return max(sentence_counts.values()) > threshold


def chinese_length(text):
    # 定义中英文标点的正则表达式
    punctuations = re.compile(r'[‘’“”、，。？！；：（）【】《》（）\[\]{}<>,.!?;:"\'\\]')
    print(len(text))
    # 去除字符串中的标点符号
    cleaned_text = re.sub(punctuations, '', text)

    # 计算处理后的字符串长度
    length = len(cleaned_text)
    print("包含纯汉字字数:", length)

    # 判断长度是否在800±tolerance范围内
    return length

# 批处理函数

import json
from tqdm import tqdm
def baseline_model(tasks):
    res = []
    num_gen = 50
    total_subtasks = len(tasks) * num_gen

    progress_bar = tqdm(total=total_subtasks, desc="Processing tasks")  # 初始化 tqdm 进度条 
    for i, task in enumerate(tasks):
        messages = [
            {"role": "system", "content": generate_prompt(i)},
            
            {"role": "user", "content": f"请编写{tasks_prompt[i]}严格要求字数为800字的有完整结尾的小说故事。"}]

        print(messages)
        # 使用 idx 生成不同的 prompt
        for n in range(num_gen):

            with torch.no_grad():            
                # 调用 API 生成小说内容
                completion = client.chat.completions.create(
                    model="GLM4-9B-lora",
                    messages=messages
                )
                response = completion.choices[0].message.content
                response = response.replace("\n", "")
                while (chinese_length(response) < 750 or chinese_length(response) >1500) or is_sentence_repetition(response):
                    print(f"######长度为{len(response)}小于700 或者大于900，#####重新生成 {len(response)}, it:{n}, task:{task}", )
                    if (is_sentence_repetition(response)):
                        print("######句子重复比率过高，重新生成######")
                    # 调用 API 生成小说内容
                    completion = client.chat.completions.create(
                                model="GLM4-9B-lora",
                                messages=messages
                            )
                    response = completion.choices[0].message.content
                    response = response.replace("\n", "")

                    if chinese_length(response) > 1500 or chinese_length(response) < 750:
                        print(f"Regenerating ", len(response))
                if chinese_length(response)<=800 and chinese_length(response) > 700:
                    response="这是一个非常有趣的故事，故事非常的精彩，请竖起耳朵，仔细聆听，且听我细细道来。"+ response
                response = response.replace("\n", "")
                res.append({
                    "instruction":"你是一个熟读各类小说的专家，请你根据要求写一段800字左右的小说。",
                    "input":task,
                    "output":response,
                })
                progress_bar.update(1)
                # 将每个结果写入文件，并添加换行符
                with open("submit.json", "a", encoding='utf-8') as file:  # 使用 'a' 模式来追加内容
                    res[-1]['instruction'] = '你是一个熟读各类小说的专家，请你根据要求写一段800字左右的小说。'
                    res[-1]['input'] = res[-1]['input'].strip()
                    res[-1]['output'] = res[-1]['output'].strip()
                    file.write(json.dumps(res[-1], ensure_ascii=False) + "\n")
    return res

# 启动批处理存为json
res_novel = baseline_model(stories)

# 查看结果
import json
from datetime import datetime

current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
# 创建新的文件名，包含时间后缀
filename = f"submit-final_{current_time}.json"
with open(filename, "w", encoding='utf-8') as file:
    for item in res_novel:
        for t in range(1):
        # 将每个元素写入文件，并添加换行符
            file.write(json.dumps(item, ensure_ascii=False) + "\n")