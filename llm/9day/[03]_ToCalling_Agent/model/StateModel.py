from typing import TypedDict, Annotated
from langchain_core.messages import AIMessage, AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langgraph.graph.message import add_messages

# 图的状态模式
class AgentState(TypedDict):
    messages:Annotated[list[AnyMessage],add_messages]