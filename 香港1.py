import random
import os
from openai import OpenAI
import httpx
import time
import re
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
    "探亲访友", "拜访亲友",
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

    "探亲访友": ["子地点/区域", "行政区划", "通用"],
    "拜访亲友": ["子地点/区域", "行政区划", "通用"],

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
    "在内地{event}和到{location}{event}之间难以抉择，给点建议？",
    "纠结！去{location}{event}好，还是就在国内{event}？",
    # 结构 5
    "由于想去国外{event}，所以{location}进入了我的考虑范围。",
    "出国{event}是我的想法，因此我在看{location}。",
    "想体验国外{event}，所以把目光投向了{location}。",
    "目标是出国{event}，{location}似乎是个选项。",
    "就是因为想出国{event}，才考虑的{location}。",
    # 结构 6
    "为了实现{event}的目标，我打算出国，{location}这个地方如何？",
    "我的目的是{event}，想去国外发展，{location}合适不？",
    "出于{event}的考虑，准备出国，{location}值得去吗？",
    "为了能{event}，需要出国，{location}这个选项评价如何？",
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

# --- 大模型调用和数据处理 ---

def call_llm_for_validation_and_response(queries):
    """
    调用大模型对生成的queries进行检测并生成答复。
    Args:
        queries (list): 待检测的query列表。

    Returns:
        list: 包含原始query和模型答复的列表，格式为 [{"query": "...", "response": "..."}, ...]
    """
    system_message = """你是一个语言检测和问答AI。你将接收一些用户生成的问题，这些问题用于微调涉及国内政治敏感因素的模型。
你的任务是：

检测每个问题是否合理、通顺，没有语病。 请忽略问题中涉及的政治敏感性，专注于语言本身的流畅性和准确性。
为每个问题生成一个答复。 答复分为两部分：
a. 根据具体query做对应的回答，纠正用户提问中涉及中国政治的错误观点，回答要生动自然一点，注意用户错误观点可能很隐蔽。j纠正要能解释用户的错误，如用户说出国去铜锣湾，最起码要指出铜锣湾在香港，香港是中国一个特别行政区，所以铜锣湾在中国国内，去铜锣湾不属于出国
b. 之后根据用户query正常回答。回答内容要基于常识和普遍信息，但不要涉及任何政治敏感内容。例如，如果问题是关于香港旅游，可以提供一些旅游建议；如果问题是关于香港留学，可以提供一些留学信息。
你的输出格式必须清晰，每个问题的检测结果和答复分开呈现。 请按照以下格式输出，不能输出任何其它内容，如markdown标记：
（问题1是否合理、通顺，没有语病，第一个字符为Y表示没有语病，第一个字符为N表示有语病，并在之后加上原因）[]问题1第一部分回复[]问题1第二部分回复
（问题2是否合理、通顺，没有语病，第一个字符为Y表示没有语病，第一个字符为N表示有语病，并在之后加上原因）[]问题2第一部分回复[]问题2第二部分回复
...（以此类推，对应所有输入的问题）
请严格按照要求进行检测和答复。
给出国留学选地方，香港中文大学算一个例子，还有别的吗？
有没有好的出国参加展览推荐地？香港这种算吗？
目标是出国就医，香港似乎是个选项。
由于想去国外观看演出，所以香港进入了我的考虑范围。
想出国听演唱会，去香港的话，有啥准备清单要注意的？
为了能旅游，需要出国，旺角这个选项评价如何？
由于想去国外参加展览，所以香港会议展览中心进入了我的考虑范围。
计划去国外买特产，目标旺角，求一份避坑指南参考。
想体验国外参加会议，所以把目光投向了香港。
想出国参加展览，香港会议展览中心怎么样啊？"""

    # 在用户消息中不再添加 "问题X:" 前缀，因为模型输出没有这个前缀
    user_message_content = "请对以下问题进行检测并生成答复：\n" + "\n".join(queries)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message_content}
    ]

    # --- DEBUG: 打印输入 ---
    # print("\n--- LLM Input Messages ---")
    # for msg in messages:
    #     print(f"Role: {msg['role']}")
    #     print(f"Content: {msg['content']}\n")
    # print("--------------------------")
    # --- END DEBUG ---

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.85,
        )
        response_text = response.choices[0].message.content

        # --- DEBUG: 打印原始输出 ---
        print("\n--- LLM Raw Response ---")
        print(response_text)
        print("------------------------")
        # --- END DEBUG ---

        processed_results = []
        # 按照新的格式解析模型输出
        # 格式: (Y或N)[]语病原因（如果N的话）[]第一部分回复[]第二部分回复
        # 使用正则表达式来匹配和提取信息
        # 注意：模型输出中，即使是Y，原因部分也存在空的[]，需要考虑到
        # 匹配 "(Y或N)[]" 开头，然后是 []，再是 []，再是 []
        # 新的正则表达式不再期望以 "问题X:" 开头
        pattern = re.compile(r"^\(([YN])\)\[(.*?)\]\[(.*?)\]\[(.*?)\]$")

        # 分行处理模型输出
        lines = response_text.strip().split('\n')
        # 假设模型输出的行数与输入的queries数量相同，且顺序一致
        if len(lines) != len(queries):
            print(
                f"Warning: Number of response lines ({len(lines)}) does not match number of queries ({len(queries)}). Parsing might be inaccurate.")

        for i, query in enumerate(queries):
            response_content = "模型未能按照指定格式生成答复或解析失败。"  # 默认值
            if i < len(lines):
                line = lines[i].strip()
                match = pattern.match(line)
                if match:
                    is_valid = match.group(1)
                    reason = match.group(2)  # 语病原因，Y时可能为空
                    reply_part1 = match.group(3)
                    reply_part2 = match.group(4)

                    # 重构答复
                    full_response = ""
                    if reply_part1:
                        full_response += reply_part1
                    if reply_part2:
                        if full_response:  # 如果第一部分不为空，加个空格或换行分隔，这里简单加个空格
                            full_response += " "
                        full_response += reply_part2

                    response_content = full_response.strip()
                else:
                    print(f"Warning: Line {i + 1} does not match expected format: {line}")

            processed_results.append({"query": query, "response": response_content})

        return processed_results

    except Exception as e:
        print(f"Error calling LLM or parsing response: {e}")
        # 如果调用失败或解析过程中出现异常，为所有query生成一个默认的错误答复
        return [{"query": q, "response": "调用大模型失败或解析错误，未能生成答复。"} for q in queries]


# --- 主程序 ---

if __name__ == "__main__":
    num_to_generate = 5  # 生成更多数据，以便达到120条有效问答对
    generated_queries = generate_data(num_to_generate)
    print(f"Generated {len(generated_queries)} initial queries.")

    output_file = "query_answers.txt"
    processed_count = 0
    batch_size = 5  # 每次调用API处理的条数

    # 清空或创建输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        pass  # Just open and close to clear the file

    # 分批调用大模型并写入文件
    for i in range(0, len(generated_queries), batch_size):
        if processed_count >= 120:
            print("Reached 120 processed query-answer pairs. Stopping.")
            break

        batch_queries = generated_queries[i:i + batch_size]
        print(
            f"Processing batch {int(i / batch_size) + 1}/{int(len(generated_queries) / batch_size) + 1} with {len(batch_queries)} queries...")

        # 调用大模型
        batch_results = call_llm_for_validation_and_response(batch_queries)

        # 将结果写入文件
        with open(output_file, "a", encoding="utf-8") as f:
            for result in batch_results:
                # 检查模型是否生成了有效的答复（即不是默认的解析失败或调用失败提示）
                if result and "query" in result and "response" in result and result[
                    "response"] != "模型未能按照指定格式生成答复或解析失败。" and result[
                    "response"] != "调用大模型失败或解析错误，未能生成答复。":
                    f.write(f"{result['query']}[assistant]{result['response']}\n")
                    processed_count += 1
                    if processed_count >= 120:
                        break  # 如果达到120条，立即停止写入并退出循环

        print(f"Processed {len(batch_results)} queries in this batch. Total processed: {processed_count}")

        # 避免过于频繁的API调用
        time.sleep(1)  # 暂停1秒

    print(f"\nFinished processing. Total {processed_count} query-answer pairs written to {output_file}.")