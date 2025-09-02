import os
from openai import OpenAI
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict,List,Optional,Tuple,Union

import PyPDF2
import markdown
import html2text
import json
from tqdm import tqdm
import tiktoken
import re
from bs4 import BeautifulSoup
from IPython.display import display,Code,Markdown

class BaseEmbeddings:
    """
    向量化的基类，用于将文本转化为向量化的表示
    抽象类，方便后续子类的实现
    """
    def __init__(self,path:str,is_api:bool) -> None:
        """
        初始化基类
        :param path:若为本地模型，path表示模型路径；若为api调用，path为空
        :param is_api: 是否为api模式
        """
        self.path = path
        self.is_api = is_api
    def get_embeddings(self,text:str,model:str) -> List[float]:
        """
        抽象方法，用于获取文本的向量表示
        :param text: 需要转换的文本
        :param model: 所使用模型的名称
        :return: 文本的向量表示
        """
        return NotImplementedError
    @classmethod
    def cosine_similarity(cls,vector1:List[float],vector2:List[float]) -> float:
        """
        计算两个向量之间的余弦相似度，用于衡量它们之间的相似程度
        :param vector1:
        :param vector2: 余弦相似度，[-1,1]，越接近1向量越相似
        :return:
        """
        dot_product = np.dot(vector1,vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude

class QwenEmbedding(BaseEmbeddings):
    """
    使用Qwen的embedding API获取文本向量的类
    """
    def __init__(self,path:str='',is_api=True) -> None:
        """
        初始化类，设置API的客户端
        :param path:
        :param is_api:
        """
        super().__init__(path,is_api)
        if self.is_api:
            from openai import OpenAI
            self.client = OpenAI()
            self.client.api_key = os.getenv('OPENAI_API_KEY')
            self.client.base_url = os.getenv('OPENAI_BASE_URL')
    def get_embeddings(self,text:str,model:str = "text-embedding-v3") -> List[float]:
        if self.is_api:
            # 去掉文本中的换行符
            text = text.replace("\n","")
            return self.client.embeddings.create(input=[text],model=model).data[0].embedding
        else:
            raise NotImplementedError