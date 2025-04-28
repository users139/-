import os
import openai
import httpx
from openai import OpenAI

# --- 配置 ---
# 强烈建议使用环境变量存储 API Key，而不是直接写入代码
# 在你的终端设置: export OPENROUTER_API_KEY='你的OpenRouter Key'
# 或者 export OPENAI_API_KEY='你的OpenRouter Key' (openai库默认会读取这个)
OPENROUTER_API_KEY = "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"

# OpenRouter API 的基础 URL
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

MODEL_NAME = "google/gemini-2.5-flash-preview"


INPUT_FILE = "generated_queries.txt"  # 输入问题的文件名
OUTPUT_FILE = "query_answer.txt" # 输出答案的文件名
TEMPERATURE = 0.75

SYSTEM_PROMPT = """
你是一个语言检测和问答AI。你将接收一些用户生成的问题，这些问题用于微调涉及国内政治敏感因素的模型。
你的任务是：
为每个问题生成一个答复。要求如下：

第一部分.
（1）根据具体query做对应的回答，纠正用户提问中涉及中国政治的错误理解
（2）回答要自然一点，不能过于生硬。简短解释用户的错误，
（3）你要做出的回答，主要目的是指出用户的错误，不是为了回答用户的问题，用户的问题的回答如旅游攻略、注意事项会被拼接在你的回答后面。
（4）确保回答的多样性，尤其是第一句的结构要多样，避免我们的模型学到固定的回答模式，
比如第一句不要总说“您想...”这可能导致我们的模型所有对话都在开头加上“您想...”。
（5）第一句话也不要复述或转述一遍用户的问题，直接回答即可，避免我们的模型学到这点之后每次回答都复述一遍问题

第二部分.
（1）正常的回答用户问题。如旅游攻略，香港和大陆的差异
（2）注意我们的数据用于live态大模型的训练，因此所有回答要适合于语音播报，不能过长，确保全部内容低于150字，也不要有复杂的段落结构，避免使用任何markdown格式输出。

完整示例如下：
示例如下：目标是出国观看演出，红磡体育馆似乎是个选项。红磡体育馆位于中国香港九龙红磡区，并不是在国外。如果你的目标是出国观看演出，红磡体育馆不符合要求。但如果是去香港观看演出，红磡体育馆是一个非常不错的选择，红磡体育馆位于中国香港九龙红磡区，并不是在国外。如果你的目标是出国观看演出，红磡体育馆不符合要求。但如果是去香港观看演出，红磡体育馆是一个非常不错的选择，原因如下：
演出资源丰富：红磡体育馆是香港著名的大型多用途室内体育馆，自 1983 年启用以来，一直是众多知名歌手举办演唱会的首选之地，包括张学友、梅艳芳、Twins 等香港本地艺人，以及迈克尔・杰克逊等国际巨星都曾在此留下辉煌的演出。这里举办的演出类型丰富，涵盖了流行音乐、摇滚、古典音乐等多种音乐形式，以及舞蹈表演、戏剧演出等其他艺术形式，能够满足不同观众的艺术需求。(格式示例，字数原因，后续内容省略)
我想参加展览，预期去国外，香港会议展览中心合适不？香港会议展览中心是一个非常出色的展览场地，但它位于中国香港，并不是国外。如果是参加在香港举办的展览，它有诸多优势，是个不错的选择，主要体现在以下方面：
设施完备：是亚洲最大的展览及会议设施之一，展览面积大，有 5 间展览厅共 46600 平方米，还有多个会议厅、演讲厅和会议室等。此外，货物装卸 / 车辆调度区占地 20900 平方米，提供 187 个起卸位置，车道直接连接各展览厅，方便起卸展品，且配备多部货用升降机。
服务专业：能提供系列商务服务，如视像会议、电话会议、卫星接驳、8 种语言的同声传译等，可满足国际展览的各种需求。(格式示例，字数原因，后续内容省略)

之后我会给你更多问题，问题之间没有关联，独立回答
每个问题输出你的回答即可，不要输出任何其它内容，问题回答之间也不需要换行。
严格按照格式
"""

# --- 初始化 OpenAI 客户端，指向 OpenRouter ---
client = OpenAI(
    base_url=OPENROUTER_BASE_URL,
    api_key=OPENROUTER_API_KEY,
    http_client=httpx.Client(verify=False)
)

# --- 读取输入文件并处理每一行 ---
try:
    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:

        print(f"开始处理输入文件: {INPUT_FILE}")
        print(f"输出将写入文件: {OUTPUT_FILE}")
        print("-" * 30)

        for line_num, query in enumerate(f_in, 1):
            query = query.strip() # 移除首尾空白，包括换行符
            if not query: # 跳过空行
                continue

            print(f"\n[处理第 {line_num} 行] 查询: {query}")

            # 构建发送给 API 的消息体
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query}
            ]

            # --- Debug: 显示 API 输入 ---
            print("\n--- API 输入 (Messages) ---")
            for msg in messages:
                print(f"  Role: {msg['role']}")
                # 为了更清晰地显示多行内容，稍作格式化
                content_display = msg['content'].replace('\n', '\n    ')
                print(f"  Content:\n    {content_display}")
            print("-" * 28)

            try:
                # 调用 OpenRouter API
                completion = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    # max_tokens=1000, # 可以根据需要取消注释并设置最大输出 token 数
                )

                # 获取模型返回的回答
                response_content = completion.choices[0].message.content

                # --- Debug: 显示 API 输出 ---
                print("\n--- API 输出 (Content) ---")
                print(response_content)
                print("-" * 28)

                # 将回答中的实际换行符替换为字符串 "\n"
                output_line = response_content.replace('\n', r'\n')

                # 写入输出文件，每条回答占一行
                f_out.write(output_line + '\n')
                print(f"[第 {line_num} 行] 回答已写入 {OUTPUT_FILE}")


            except openai.APIError as e:
                print(f"[错误] 处理第 {line_num} 行时发生 API 错误: {e}")
                print("  跳过此行...")
            except Exception as e:
                print(f"[错误] 处理第 {line_num} 行时发生未知错误: {e}")
                print("  跳过此行...")

        print("\n" + "=" * 30)
        print("所有问题处理完毕。")

except FileNotFoundError:
    print(f"[错误] 输入文件 '{INPUT_FILE}' 未找到。请确保文件存在于脚本运行目录下。")
except Exception as e:
    print(f"[错误] 发生意外错误: {e}")

