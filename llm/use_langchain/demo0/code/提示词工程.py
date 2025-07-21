from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

"""
聊天模型，有三种角色类型：
    助手（Assistant）消息指当前的消息是AI回答的内容
    人类（user）消息指发送给AI的内容
    系统（system）消息用来描述AI身份
"""
# 例1
chat_template = ChatPromptTemplate.from_messages(
    [
        ("system","你是一位人工智能助手，你的名字是{name}。"),
        ("human","你好"),
        ("ai","我很好，谢谢"),
        ("human","{user_input}"),
    ]
)

messages = chat_template.format_messages(name="Bob",user_input="你的名字是什么？")
print(messages)

# 例2
from langchain.prompts import HumanMessagePromptTemplate
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate

chat_template = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content=(
                "你是一个乐于助人的助手，可以润色内容，使其看起来更简单易读"
            )
        ),
        HumanMessagePromptTemplate.from_template("{text}"),
    ]
)

messages = chat_template.format_messages(text="我不喜欢吃好吃的东西")
print(messages)

# 例3，可以传入一个上下文消息的消息列表，可以初始化几个问题，做初始化角色定义之类
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,SystemMessage

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","你是一个乐于助人的助手"),
        MessagesPlaceholder("msgs"),
    ]
)

result = prompt_template.invoke({"msgs":[HumanMessage(content="hi"),HumanMessage(content="hello")]})
print(result)

# 提示词追加：