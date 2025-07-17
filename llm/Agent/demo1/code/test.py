import lazyllm
from lazyllm import Document,Retriever
chat = lazyllm.OnlineChatModule(
    source="deepseek",
    model='deepseek-chat',
    api_key="sk-c50f7fa6d53844819f14d0928cba58d9"
)
# lazyllm.WebModule(chat,port=(23466,23470)).start().wait()
doc = Document("/data")
separator = '\n' + '='*200 + '\n'
retriever = Retriever(
    doc,
    group_name=Document.CoarseChunk,
    similarity="bm25_chinese",
    topk=3,
    output_format="content",
    join=separator
)

res = retriever("高速摄影的代价")
print(res)