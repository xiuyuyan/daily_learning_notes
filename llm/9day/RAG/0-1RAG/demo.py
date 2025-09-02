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

client = OpenAI(
    # model = 'qwq-32b',
    base_url = 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    api_key='sk-b053451148f44f6fa179583f03bcfed2',
)

os.environ["OPENAI_API_KEY"] = 'sk-b053451148f44f6fa179583f03bcfed2'
os.environ["OPENAI_BASE_URL"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# 文本向量化模块
response = client.embeddings.create(
    input="测试文本",
    model='text-embedding-v3'
)

print(response.data[0].embedding)