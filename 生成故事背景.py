import os
from openai import OpenAI
import httpx
import textwrap

# 获取 OpenRouter API Key (建议使用环境变量)
# OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")
# if not OPENROUTER_API_KEY:
#     print("请设置 OPENROUTER_API_KEY 环境变量")
#     exit()

# 为了演示方便，直接在这里设置，请在实际应用中替换
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


def get_llm_response_guided(messages):
    """
    调用大模型获取回复，并引导其给出问题和选项
    """
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.8, # 稍微提高温度，增加生成的多样性
            max_tokens=800 # 增加最大 token 长度，以便容纳更多问题和选项
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"调用大模型时发生错误: {e}")
        return None

def build_story_background_guided():
    """
    通过引导式与大模型交互构建故事背景
    """
    print("欢迎来到互动式末日生存故事背景构建器！")
    print("我们将通过对话和选项来共同创造一个末日世界和你的角色设定。")
    print("-" * 30)

    # 更新 System Prompt，明确指导大模型的行为
    system_prompt = textwrap.dedent("""
    你是一个互动式故事背景构建助手。你的任务是**仅限于**与用户共同完善末日生存故事的背景设定，**不要**开始叙述故事的开端或具体的剧情发展。

    在每次回复中，请：
    1. 根据之前的对话和用户输入，提出 1 到 3 个关于**末日世界的规则、主角的初始状态、环境的特点**等背景的关键问题。
    2. 为每个问题提供 2 到 3 个建议的选项。请使用编号（如 1., 2., 3.）来标记选项。
    3. **不要**在回复中包含任何描述主角行动结果或剧情推进的叙事内容。
    4. 鼓励用户回答问题，选择选项（例如输入选项编号），或者提供他们自己的想法和补充。
    5. 保持回复的连贯性，并根据用户的输入调整后续的问题，但始终聚焦于背景构建。
    6. 当用户输入 '结束' 时，你需要总结整个对话中构建的末日世界背景、主角信息和当前环境的关键设定。
    """)

    messages = [
        {"role": "system", "content": system_prompt}
    ]

    # 初始化对话，提出第一个问题和选项
    initial_question_prompt = textwrap.dedent("""
    让我们从最开始的地方说起。末日是如何发生的？你认为最可能的原因是什么？

    1.  一场致命的病毒爆发，将大部分人类变成了行尸走肉。
    2.  全球性的核战争，导致文明崩溃和环境污染。
    3.  未知的环境灾难，例如极端气候变化或外星入侵。
    """)
    print(initial_question_prompt)
    messages.append({"role": "assistant", "content": initial_question_prompt}) # 将初始问题也添加到对话历史

    # 循环进行对话，直到用户输入 '结束'
    print("输入 '结束' 来完成背景构建。")
    while True:
        user_input = input("\n你的输入 (可以直接回答问题，输入选项编号，或自由补充): ")
        if user_input.lower() == '结束':
            break

        messages.append({"role": "user", "content": user_input})

        # 调用大模型获取回复
        llm_response = get_llm_response_guided(messages)

        if llm_response:
            print("\n大模型回复:")
            print(textwrap.fill(llm_response, width=80)) # 格式化输出
            messages.append({"role": "assistant", "content": llm_response})
        else:
            print("未能获取大模型回复，请稍后再试。")

    # 提炼最终的背景 Prompts
    print("-" * 30)
    print("背景构建完成。正在提炼最终的背景 Prompts...")

    # 构建用于提炼 Prompts 的指令
    # 这个指令将在用户输入 '结束' 后发送给大模型
    summary_prompt = textwrap.dedent("""
    根据我们之前的对话，请总结出末日世界的背景、主角信息以及当前所处环境的关键设定。请以清晰、结构化的文本格式输出，例如：

    [末日世界背景]
    末日类型：...
    世界状态：...
    主要威胁：...
    资源情况：...

    [主角信息]
    姓名：...
    职业：...
    特质：...
    目标：...

    [当前环境]
    位置：...
    环境描述：...
    当前困境：...

    请只输出总结内容，不要包含额外的对话或说明。
    """)

    messages.append({"role": "user", "content": summary_prompt})

    final_background_prompts = get_llm_response_guided(messages)

    if final_background_prompts:
        print("\n最终背景 Prompts:")
        print(final_background_prompts)
        return final_background_prompts
    else:
        print("未能生成最终背景 Prompts。")
        return None

if __name__ == "__main__":
    background_prompts = build_story_background_guided()
    if background_prompts:
        # 你可以将这个 final_background_prompts 保存到文件中，供后续使用
        # 例如：
        with open("story_background_guided.txt", "w", encoding="utf-8") as f:
             f.write(background_prompts)
        print("\n背景 Prompts 已生成。")