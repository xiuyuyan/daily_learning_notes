from data_process.database import create_and_populate_database,query_by_product_name
from tools.func import tools
from openai import OpenAI
import json

def question():
    while True:
        messages = []
        prompt = input("\n请您提出问题：")
        if prompt.lower() == "退出":
            break
        # 添加提问到消息列表
        messages.append({'role':'user','content':prompt})

        # 检查是否需要使用外部函数
        completion = client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=tools,
            parallel_tool_calls=False, # 是否允许并行，默认为True
        )

        # 提取回答内容
        response = completion.choices[0].message
        tool_calls = completion.choices[0].message.tool_calls

        # 处理外部函数调用
        if tool_calls:
            function_name = tool_calls[0].function.name # 调用的外部函数的名称
            function_args = json.loads(tool_calls[0].function.arguments) # 外部函数必须的参数

            function_response = available_functions[function_name](**function_args)

            messages.append(response)

            messages.append(
                {
                    "role": "tool",
                    "name": function_name,
                    "content": str(function_response),
                    "tool_call_id": tool_calls[0].id,
                }
            )

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

    available_functions = {"query_by_product_name": query_by_product_name}

    question()