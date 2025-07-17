from langchain_openai import ChatOpenAI

model = ChatOpenAI(
    model = 'deepseek-chat',
    base_url = 'https://api.deepseek.com/v1',
    api_key='sk-40aac449e559480b9676b129ff8a20dc',
)

"""
# (同步)流式传输，用于输出很长时，防止用户等待时间过久，所以流式分块输出
chunks = []
for chunk in model.stream("天空是什么颜色？"):
    chunks.append(chunk)
    print(chunk.content,end="|",flush=True)
"""

# （异步）
import asyncio # 为了支持异步调用
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser,JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

prompts = ChatPromptTemplate.from_template("请给我讲一个关于{topic}的笑话")
llm = ChatOpenAI(
    model = 'deepseek-chat',
    base_url = 'https://api.deepseek.com/v1',
    api_key='sk-40aac449e559480b9676b129ff8a20dc',
)
output_parser = StrOutputParser()
chain = prompts | llm | output_parser

async def async_steram1():
    async for chunk in chain.astream({"topic":"猫"}):
        print(chunk,end="|",flush=True)
async def async_steram2():
    async for chunk in chain.astream({"topic":"狗"}):
        print(chunk,end="|",flush=True)

asyncio.run(asyncio.gather(async_steram1(), async_steram2()))