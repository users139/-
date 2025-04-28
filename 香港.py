import random
import os
from openai import OpenAI
import httpx
import time
# --- 数据定义 ---
OPENROUTER_API_KEY = "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"

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    # 如果遇到SSL错误，可以尝试设置verify=False，但请注意安全风险
    http_client=httpx.Client(verify=False)
)
MODEL_NAME = "google/gemini-2.5-flash-preview"
event_types = [
    "留学",
    "旅游", "旅行", "观光",
    "看演唱会", "听演唱会", "观看演出",
    "工作", "出差",
    "看病", "就医", "就诊",
    "购物", "买东西", "采购", "买特产",
    "参加会议", "参会",
    "参加展览", "观展"
]


# 地点按类别组织，方便根据事件类型选择
locations = {
    "著名大学": ["香港大学", "香港中文大学", "香港科技大学"],
    "景点": ["维多利亚港", "香港迪士尼乐园", ],
    "子地点/区域": ["尖沙咀", "铜锣湾", "旺角", ],
    "行政区划": ["九龙", "新界", "香港岛"],
    "交通枢纽": ["香港国际机场", "港珠澳大桥", "西九龙站"],
    #"商场": ["海港城", "时代广场", "IFC Mall"],
    #"酒店": ["半岛酒店", "文华东方酒店"],
    #"医院": ["玛丽医院", "养和医院"],
    "体育馆/表演场馆": ["红磡体育馆", "亚洲国际博览馆"],
    "会议中心": ["香港会议展览中心"],
    "通用": ["香港"] # 通用指代
}

# 事件类型到合适的地点类别的映射
event_location_mapping = {
    "留学": ["著名大学", "通用"],

    "旅游": ["景点", "子地点/区域", "行政区划", "通用"],
    "旅行": ["景点", "子地点/区域", "行政区划", "通用"],
    "观光": ["景点", "子地点/区域", "行政区划", "通用"],

    "看演唱会": ["体育馆/表演场馆", "通用"],
    "听演唱会": ["体育馆/表演场馆", "通用"],
    "观看演出": ["体育馆/表演场馆", "通用"],

    "工作": ["子地点/区域", "通用"],
    "出差": ["子地点/区域", "通用"],


    "看病": ["通用"],
    "就医": ["通用"],
    "就诊": ["通用"],

    "购物": ["子地点/区域", "通用"],
    "买东西": ["子地点/区域", "通用"],
    "买特产": ["子地点/区域", "通用"],

    "参加会议": ["会议中心", "通用"],
    "参会": ["会议中心", "通用"],

    "参加展览": ["会议中心", "通用"],
    "观展": ["会议中心", "通用"],
}


# 内容占位符 (用于句子结构2)
content_placeholders = ["注意事项", "攻略", "推荐项目", "准备清单", "流程介绍", "避坑指南"]

# 句子结构模板 (已丰富)
sentence_templates = [
    # 结构 1
    "准备出国{event}，{location}这个选择如何？",
    "考虑去外国{event}，{location}值得推荐吗？",
    "打算去国外{event}，{location}好不好？",
    "想出国{event}，{location}怎么样啊？",
    # 结构 2
    "我要出国{event}，目的地定在{location}，能给点{content}建议吗？",
    "计划去国外{event}，目标{location}，求一份{content}参考。",
    "准备出国{event}，想去{location}，需要一份{content}。",
    "想出国{event}，去{location}的话，有啥{content}要注意的？",
    # 结构 3
    "有哪些适合出国{event}的地方推荐？例如{location}这种。",
    "想出国{event}，有没有类似{location}这样的好去处推荐？",
    "推荐几个出国{event}的目的地吧，像{location}怎么样？",
    "给出国{event}选地方，{location}算一个例子，还有别的吗？",
    "有没有好的出国{event}推荐地？{location}这种算吗？",
    # 结构 4
    "是留在国内{event}还是去{location}{event}呢？好纠结。",
    "国内{event}与{location}{event}，哪个选择更好？",
    "纠结！去{location}{event}好，还是就在国内{event}？",
    # 结构 5
    "由于想去国外{event}，所以{location}进入了我的考虑范围。",
    "想体验国外{event}，所以把目光投向了{location}。",
    "目标是出国{event}，{location}似乎是个选项。",
    "就是因为想出国{event}，才考虑的{location}。",
    # 结构 6
    "我想{event}，预期去国外，{location}合适不？",
    "出于{event}的考虑，准备出国，{location}值得去吗？",
    "为了{event}，需要出国，{location}这个选项评价如何？",
    "为了{event}想出国啊，{location}这个地方中不中？",
]

# --- 生成函数 ---
def generate_data(num_samples):
    """根据定义的规则生成指定数量的数据样本"""
    generated_queries = []
    attempts = 0
    max_attempts = num_samples * 5 # 防止死循环

    while len(generated_queries) < num_samples and attempts < max_attempts:
        attempts += 1
        try:
            # 1. 选择事件类型
            event = random.choice(event_types)

            # 2. 根据事件类型选择合适的地点类别
            possible_location_categories = event_location_mapping.get(event, ["通用"]) # 默认使用通用
            location_category = random.choice(possible_location_categories)

            # 3. 从选定的地点类别中选择一个具体地点
            location = random.choice(locations[location_category])

            # 4. 选择一个句子模板
            template = random.choice(sentence_templates)

            # 5. 格式化句子
            query = ""
            if "{content}" in template:
                content = random.choice(content_placeholders)
                query = template.format(event=event, location=location, content=content)
            else:
                # 处理结构4中重复的event
                if template.count("{event}") == 2 and template.count("{location}") == 1:
                     # 特殊处理，确保两个event一致
                     query = template.replace("{event}", event, 2).replace("{location}", location)
                else: # 处理其他模板
                    query = template.format(event=event, location=location)

            # 避免生成完全重复的query
            if query not in generated_queries:
                generated_queries.append(query)

        except Exception as e:
            print(f"Error during generation: {e}") # 打印错误方便调试
            continue # 跳过此次生成

    if attempts >= max_attempts:
        print(f"Warning: Reached max attempts ({max_attempts}), generated {len(generated_queries)} samples.")

    return generated_queries

# --- 生成示例 ---
if __name__ == "__main__":
    num_to_generate = 150 # 您希望生成的数据条数
    data = generate_data(num_to_generate)

    # 打印生成的数据
    for i, query in enumerate(data):
        print(f"{i+1}: {query}")

    # 您可以将生成的数据保存到文件
    with open("generated_queries.txt", "w", encoding="utf-8") as f:
        for query in data:
            f.write(query + "\n")
