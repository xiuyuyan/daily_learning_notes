import os
import sys
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
from .embedding import BaseEmbeddings,QwenEmbedding
from .load_chunk import ReadFiles,Documents

class VectorStore:
    def __init__(self,document:List[str] = None) -> None:
        """
        初始化向量存储类，存储文档和对应的向量表示
        :param document: 文档列表
        """
        if document is None:
            document = []
        self.document = document
        self.vectors = [] # 存储文档的向量表示
    def get_vector(self,EmbeddingModel:BaseEmbeddings):
        """
        使用传入的Embedding模型将文档向量化
        :param EmbeddingModel: 传入的模型（需继承自BaseEmbeddings）
        :return: 文档对应的向量列表
        """
        self.vectors = [EmbeddingModel.get_embeddings(doc) for doc in self.document] # 遍历
        return self.vectors

    def parsist(self,path:str='storage'):
        """
        将文档和对应的向量表示持久化存储到本地目录中，以便后续使用
        :param path: 存储路径
        :return:
        """
        if not os.path.exists(path):
            os.makedirs(path)
        np.save(os.path.join(path,'vectors.npy'),self.vectors) # 保存向量为np文件
        with open(os.path.join(path,'documents.txt'),'w') as f:
            for doc in self.document:
                f.write(f"{doc}\n")

    def load_vector(self,path:str='storage'):
        """
        从本地加载之前保存的文档和向量
        :param path:
        :return:
        """
        self.vectors = np.load(os.path.join(path,'vectors.npy')).tolist()
        with open(os.path.join(path,'documents.txt'),'r') as f:
            self.document = [line.strip() for line in f.readlines()]

    def get_similarity(self,vector1:List[float],vector2:List[float]) -> float:
        """
        计算两个向量的余弦相似度
        :param vector1:
        :param vector2:
        :return: 余弦相似度
        """
        dot_product = np.dot(vector1,vector2)
        magnitude = np.linalg.norm(vector1) * np.linalg.norm(vector2)
        if not magnitude:
            return 0
        return dot_product / magnitude

    def query(self,query:str,EmbeddingModel:BaseEmbeddings,k:int=1) -> List[str]:
        """
        根据查询文本，检索最相关的k个文档片段
        :param query: 查询文本
        :param EmbeddingModel:
        :param k:
        :return: 返回最相似的文档列表
        """
        query_vector = EmbeddingModel.get_embeddings(query)

        similarities = [self.get_similarity(query_vector,vector) for vector in self.vectors]

        top_k_indices = np.argsort(similarities)[-k:][::-1]

        return [self.document[idx] for idx in top_k_indices]