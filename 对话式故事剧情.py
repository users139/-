import os
from openai import OpenAI
import httpx
import textwrap
import json # 使用json来解析结构化输出可能更健壮
import re

OPENROUTER_API_KEY = ""

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
    http_client=httpx.Client(verify=False)  # 如果遇到SSL错误，可以尝试设置verify=False，但请注意安全风险
)

# 选择一个 OpenRouter 支持的模型
# 请查看 OpenRouter 文档以获取可用模型列表
# 示例模型名称，请根据实际情况修改
MODEL_NAME = "google/gemini-2.5-flash-preview"  # 示例模型

DEBUG_MODE = True # 设置为 True 开启调试信息，设置为 False 关闭


def get_llm_structured_response(messages):
    """
    调用大模型获取结构化回复（剧情、选项、结果）
    增加调试信息
    """
    if DEBUG_MODE:
        print("\n" + "="*40 + " DEBUG: 发送给大模型的 Prompts " + "="*40)
        # 打印messages，为了清晰，可以使用json.dumps进行格式化
        print(json.dumps(messages, indent=2, ensure_ascii=False))
        print("="*90 + "\n")

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.85,
        )

        raw_response_text = response.choices[0].message.content.strip()

        if DEBUG_MODE:
            print("\n" + "="*40 + " DEBUG: 接收到的大模型原始回复 " + "="*40)
            print(raw_response_text)
            print("="*90 + "\n")

        return raw_response_text

    except Exception as e:
        print(f"调用大模型时发生错误: {e}")
        return None

def parse_structured_response(response_text):
    """
    解析大模型生成的结构化文本，提取剧情、选项和结果
    假设大模型按照以下JSON格式输出，可能包含markdown的```json```包裹：
    {
        "剧情": "...",
        "选项": [
            {"描述": "...", "结果类型": "...", "结果描述": "..."},
            ...
        ]
    }
    """
    # 尝试查找并提取```json```标记内的JSON字符串
    json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
    if json_match:
        json_string = json_match.group(1)
    else:
        # 如果没有```json```标记，则尝试直接解析整个文本为JSON
        json_string = response_text

    try:
        data = json.loads(json_string)
        return data.get("剧情"), data.get("选项")
    except json.JSONDecodeError as e:
        print(f"解析大模型输出失败: {e}")
        print("原始输出:")
        print(response_text)
        return None, None


system_prompt = """
你是一个末日生存游戏的剧情讲述者，你阅读过无数本相关小说、影视作品、电子游戏等，能够创造出各种引人入胜的精彩剧情。
根据背景和玩家的选择，生成剧情，并提供四个可选的行动选项。
每个选项需要包含描述、结果类型（死亡、获益、获弊、无事发生）和结果描述,
结果描述是玩家选择了这一选项后剧情的走向，而不是可能的结果，要尽可能丰富，能够改变剧情走向推动剧情发展，不能是“可能导致遭到袭击”这样模糊的描述。
注意用户只能看到每个选项，不能看到选项背后的结果，你要适当在后续中描述这一结果。
请以JSON格式输出，包含'剧情'和'选项'两个键。
注意之后你输出的选项不会被放到下一轮输出的上下文，相反只有用户的选择和结果。
"""

def play_interactive_story(background_prompts):
    """
    进行互动式故事游戏
    """
    print("开始末日生存之旅...")
    print("-" * 30)

    # 初始化对话历史，包含背景 Prompts
    messages = [
        {"role": "system", "content": system_prompt}
    ]
    # 将背景 Prompts 添加到对话历史中
    messages.append({"role": "user", "content": f"故事背景：\n{background_prompts}\n\n请开始故事，并提供初始的行动选项。"})


    while True:
        # 调用大模型生成剧情和选项
        structured_response_text = get_llm_structured_response(messages)

        if not structured_response_text:
            print("未能获取大模型回复，游戏结束。")
            break

        # 解析大模型的输出
        current_plot, options = parse_structured_response(structured_response_text)

        if not current_plot or not options or len(options) != 4:
            print("大模型输出格式不正确或选项数量不足，游戏可能无法继续。")
            print("原始输出:")
            print(structured_response_text)
            # 可以选择在这里结束游戏或尝试重新生成
            break

        # --- 重要修改：将大模型生成的剧情和选项添加到对话历史中 ---
        # 将大模型生成的整个结构化输出作为 assistant 的回应添加到历史中
        # 这样大模型在下一轮就能看到它自己生成的完整剧情和选项信息
        messages.append({"role": "assistant", "content": structured_response_text})
        # 也可以选择只添加剧情，但添加整个结构化输出可以让大模型更清楚地理解上一轮的“选项”和“结果”是如何关联的

        # 展示剧情
        print("\n[剧情]")
        print(textwrap.fill(current_plot, width=80))

        # 展示选项 (只展示描述)
        print("\n[选项]")
        for i, option in enumerate(options):
            print(f"{i + 1}. {option['描述']}")

        # 获取玩家选择
        while True:
            try:
                choice = int(input("\n请选择你的行动 (输入数字 1-4): "))
                if 1 <= choice <= 4:
                    break
                else:
                    print("无效的输入，请输入 1-4 之间的数字。")
            except ValueError:
                print("无效的输入，请输入一个数字。")

        # 处理玩家选择
        selected_option = options[choice - 1]
        result_type = selected_option["结果类型"]
        result_description = selected_option["结果描述"]
        selected_option_description = selected_option["描述"] # 保存选项描述用于后续Prompts

        print(f"\n你选择了: {selected_option_description}")

        # 根据结果类型判断是否结束游戏
        if result_type == "死亡":
            print("\n[游戏结束]")
            print(textwrap.fill(result_description, width=80))
            print("你的末日生存之旅在此画上了句号。")
            break
        else:
            # --- 重要修改：将玩家的选择和结果添加到对话历史中 ---
            # 将玩家的选择和结果作为用户的回应添加到历史中
            # 这样大模型就知道玩家在上一步做了什么，以及导致了什么结果
            messages.append({"role": "user", "content": f"我选择了选项 {choice}：{selected_option_description}。结果是：{result_description}"})

            # 在下一轮循环中，大模型会根据完整的对话历史（包括它自己生成的剧情和玩家的选择及结果）生成后续剧情和选项



# --- 主程序 ---
if __name__ == "__main__":
    # 加载第一阶段生成的背景 Prompts
    background_file = "story_background_guided.txt" # 假设背景 Prompts 保存在这个文件中
    try:
        with open(background_file, "r", encoding="utf-8") as f:
            story_background_prompts = f.read()
        print(f"成功加载背景文件: {background_file}")
    except FileNotFoundError:
        print(f"错误：未找到背景文件 '{background_file}'。请先运行背景构建脚本。")
        exit()

    play_interactive_story(story_background_prompts)
