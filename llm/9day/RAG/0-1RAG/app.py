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
from blocks.embedding import BaseEmbeddings,QwenEmbedding
from blocks.load_chunk import ReadFiles,Documents
from blocks.dataset import VectorStore
from llm_model.llm_model import deepseekChat
def run(question:str,knowledge_base_path:str,k:int=1) -> str:
    """
    运行基于RAG的问答助手
    :param question: 问题
    :param knowledge_base_path:知识库的路径
    :param k:选取与问题最相关的k个文档片段，默认为1
    :return:
    """
    os.environ["OPENAI_API_KEY"] = 'sk-b053451148f44f6fa179583f03bcfed2'
    os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

    # 1.加载切分文档
    docs = ReadFiles(knowledge_base_path).get_content(max_token_len=600,cover_content=150)
    vector = VectorStore(docs)

    # 2.embedding，将知识库内容向量化
    embedding = QwenEmbedding()
    vector.get_vector(EmbeddingModel=embedding)

    # 3.将向量化的文档和原文档保存到本地（可选）
    vector.parsist(path='storage')

    # 4.在数据库中检索最相关的文档片段
    content = vector.query(question,EmbeddingModel=embedding,k=k)[0]

    # 使用deepseek模型生成答案
    chat = deepseekChat()
    answer = chat.chat(question,[],content)

    return answer

if __name__ == "__main__":
    question = "演唱会内场和看台的区别？"
    knowledge_base_path = "./data"

    ans = run(question,knowledge_base_path,3)

    print(ans)

