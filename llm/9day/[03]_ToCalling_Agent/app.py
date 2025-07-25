import os
import getpass
from langchain_openai import OpenAI, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from model.Structed_data import *
from tools.func import fetch_real_time_info,get_weather,insert_db
from langgraph.graph import START, StateGraph, END
from langgraph.graph import StateGraph
from IPython.display import Image, display
from langchain_core.messages import AIMessage,AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages
from model.StateModel import AgentState

# 定义节点函数
def chat_with_model(state):
    """generate structured output"""
    print(state)
    print("-----------------")
    messages = state['messages'][-1]
    structed_llm = llm.with_structured_output(FinalResponse)
    response = structed_llm.invoke(messages)
    return {"messages":[response]}
def final_answer(state):
    """generate natural language response"""
    print(state)
    print("-----------------")
    messages = state['messages'][-1]
    response = messages.final_output.response
    return {"messages": [response]}
def execute_function(state):
    print(state)
    print("-----------------")
    messages = state['messages'][-1].final_output
    response = tool_node.invoke({"messages":[model_with_tools.invoke(str(messages))]})
    print(response)
    response = response['messages'][0].content
    return {"messages":[response]}

# 定义路由函数
def generate_branch(state:AgentState):
    result = state['messages'][-1]
    output = result.final_output

    if isinstance(output,ConversationalResponse):
        return False
    else:
        return True

if __name__ == '__main__':
    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key="sk-40aac449e559480b9676b129ff8a20dc",
    )

    tools = [fetch_real_time_info,get_weather,insert_db]
    tool_node = ToolNode(tools)
    model_with_tools = llm.bind_tools(tools)

    # 构建图
    graph = StateGraph(AgentState)
    # 添加三个节点
    graph.add_node("chat_with_model",chat_with_model)
    graph.add_node("final_answer",final_answer)
    graph.add_node("execute_function",execute_function)
    # 设置图的启动节点
    graph.set_entry_point("chat_with_model")
    # 设置条件边
    graph.add_conditional_edges(
        "chat_with_model",
        generate_branch,
        {
            True:"execute_function",
            False:"final_answer",
        }
    )
    # 设置终止节点
    graph.set_finish_point("final_answer")
    graph.set_finish_point("execute_function")
    # 编译
    graph = graph.compile()
    with open("graph.png", "wb") as f:
        f.write(graph.get_graph(xray=True).draw_mermaid_png())
