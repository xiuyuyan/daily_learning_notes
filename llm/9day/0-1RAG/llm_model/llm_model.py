from langchain.chains.flare.prompts import PROMPT_TEMPLATE
from openai import OpenAI
from typing import List

from .my_prompt import PROMPT_TEMPLATE

class BaseModel:
    """
    基础模型类，作为所有模型的基类
    包含一些通用的接口，如加载模型、生成回答等
    """
    def __init__(self,path:str = '') -> None:
        self.path = path
    def chat(self,prompt:str,history:List[dict],content:str) -> str:
        """
        使用模型生成回答的抽象方法
        :param prompt: 提问内容
        :param history: 之前的对话历史
        :param content: 提供的上下文内容
        :return: 回答
        """
        pass
    def load_model(self):
        pass

class deepseekChat(BaseModel):
    """
    基于deepseek模型的对话类
    """
    def __init__(self,api_key:str="sk-40aac449e559480b9676b129ff8a20dc",base_url:str="https://api.deepseek.com/v1",type:str="deepseek-chat") -> None:
        """
        初始化模型
        :param api_key:
        :param base_url:
        :param type: 模型类型,deepseek-chat/deepseek-reasonor
        """
        super().__init__()
        self.type = type
        self.client = OpenAI(
            # model=type,
            base_url=base_url,
            api_key=api_key,
        )
    def chat(self,prompt:str,history:List=[],content:str='') -> str:
        """
        调用deepseek生成回答
        :param prompt: 提示词
        :param history: 之前的对话历史（可选）
        :param content: 可参考的上下文信息（可选）
        :return: 回答
        """
        full_prompt = PROMPT_TEMPLATE['deepseek_prompt'].format(question=prompt,
                                                                context=content)

        response = self.client.chat.completions.create(
            model=self.type,
            messages=[
                {
                    "role":"user",
                    "content":full_prompt,
                }
            ]
        )

        return response.choices[0].message.content
