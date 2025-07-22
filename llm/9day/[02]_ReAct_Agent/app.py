import json

import requests
import os
import re
from dotenv import load_dotenv
from openai import OpenAI

from blocks.agent import CustomerSeriviceAgent
from tools.func import calculate,read_store_promotions,query_by_product_name
from tools.func_desc import *

def load_config():
    with open('config.json', 'r') as f:
        return json.load(f)

def get_client(config):
    if config['deepseek'].get('use_model',True):
        return OpenAI(
            base_url = 'https://api.deepseek.com/v1',
            api_key = os.environ.get("API_KEY"),
        )
    else:
        pass

def get_max_iterations(config):
    # 选择最大迭代次数
    if config['deepseek'].get('use_model',True):
        return config['deepseek']['max_iterations']
    else:
        return 10

def run():
    # 导入配置json文件
    config = load_config()
    try:
        # 获取服务端实例，本项目使用deepseek API
        client = get_client(config)
        # 实例化agent
        agent = CustomerSeriviceAgent(client,config)
    except Exception as e:
        print(f"Error initializing the AI client: {str(e)}")
        print("Please check your configuration and ensure the AI service is running.")
        return

    tools = {
        "query_by_product_name": query_by_product_name,
        "read_store_promotions": read_store_promotions,
        "calculate": calculate,
    }

    # 主循环：多次输入
    while True:
        query = input("请输入您的问题或输入'退出'结束聊天：")
        if query == '退出':
            break

        iteration = 0
        max_iterations = get_max_iterations(config)
        while iteration < max_iterations: # 处理每一个query
            try:
                result = agent(query)
                print(result)
                action_re = re.compile('^Action: (\w+): (.*)$')
                action = [action_re.match(a) for a in result.split('\n') if action_re.match(a)]
                if action:
                    action_parts = result.split("Action:",1)[1].strip().split(": ",1)
                    tool_name = action_parts[0]
                    tool_args = action_parts[1] if len(action_parts) > 1 else ""
                    if tool_name in tools:
                        try:
                            observation = tools[tool_name](tool_args)
                            query = f"Observation: {observation}"
                        except Exception as e:
                            query = f"Observation: Error occurred while executing the tool: {str(e)}"
                    else:
                        query = f"Observation: Tool '{tool_name}' not found"
                elif "Answer:" in result:
                    print(f"客服回复：{result.split('Answer:', 1)[1].strip()}")
                    break
                else:
                    query = "Observation: No valid action or answer found. Please provide a clear action or answer."
            except Exception as e:
                print(f"An error occurred while processing the query: {str(e)}")
                print("Please check your configuration and ensure the AI service is running.")
                break
            iteration = iteration + 1

        if iteration == max_iterations:
            print("Reached maximum number of iterations without a final answer.")

if __name__ == "__main__":
    load_dotenv()
    run()
