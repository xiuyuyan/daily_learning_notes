from data_process.database import *
from tools.func import tools
from openai import OpenAI
import json

def question():
    while True:
        messages = []
        prompt = input('\n提出一个问题： ')
        if prompt.lower() == "退出":
            break  # 如果输入的是“退出”，则结束循环

        # 添加用户的提问到消息列表
        messages.append({'role': 'user', 'content': prompt})

        # 检查是否需要调用外部函数
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            parallel_tool_calls=True  # 是否允许并行函数调用，默认为True
        )

        # 提取回答内容
        response = completion.choices[0].message
        tool_calls = completion.choices[0].message.tool_calls

        if tool_calls:
            messages.append(completion.choices[0].message) # 添加tool_calls_id
            # 处理外部函数调用
            for tool_call in tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)
                # 执行外部函数获得函数返回值
                function_response = available_functions[function_name](**function_args)
                # 将外部函数的返回值结合tool_calls_id添加到message
                messages.append({
                    "role":"tool",
                    "content":str(function_response),
                    "tool_call_id":tool_call.id,
                })
            # 调用llm结合function call的返回值进行再次生成
            second_response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
            )
            # 获取最终结果
            final_response = second_response.choices[0].message.content
            messages.append({'role': 'assistant', 'content': final_response})
            print(final_response)
        else:
            # 打印响应并添加到消息列表
            print(response.content)
            messages.append({'role': 'assistant', 'content': response.content})

if __name__ == '__main__':

    # 创建数据库并mock部分数据
    create_and_populate_database()

    client = OpenAI(
        base_url = 'https://api.deepseek.com/v1',
        api_key = 'sk-40aac449e559480b9676b129ff8a20dc',
    )

    available_functions = {
        "query_by_product_name": query_by_product_name,
        "read_store_promotions": read_store_promotions
    }

    question()