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
