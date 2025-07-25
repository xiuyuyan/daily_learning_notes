from langchain_core.tools import tool
from model.Structed_data import SearchQuery,UserInfo,WeatherLoc
import json,requests
from data_process.data_process import *

@tool(args_schema=SearchQuery) # 指定参数类型为SearchQuery
def fetch_real_time_info(query):
    """Get real-time Internet information"""
    url = "https://google.serper.dev/search"
    payload = json.dumps({
        "q": query,
        "num":1,
    })
    headers = {
        "X-API-KEY":"abc4ee8edcfe4464ec51e359c0a68f2124f19750",
        "Content-Type":"application/json"
    }
    response = requests.post(url, data=payload, headers=headers)
    data = json.loads(response.text) # 将返回的字符串转换为字典
    if 'organic' in data:
        return json.dumps(data['organic'], ensure_ascii=False)  # 返回'organic'部分的JSON字符串
    else:
        return json.dumps({"error": "No organic results found"}, ensure_ascii=False)  # 如果没有'organic'键，返回错误信息

@tool(args_schema=WeatherLoc)
def get_weather(location):
    """Call to get the current weather."""
    if location.lower() in ["北京"]:
        return "北京的温度是16度，天气晴朗。"
    elif location.lower() in ["上海"]:
        return "上海的温度是20度，部分多云。"
    else:
        return "不好意思，并未查询到具体的天气信息。"

@tool(args_schema=UserInfo)
def insert_db(name,age,email,phone):
    """Insert user information into the database, The required parameters are name, age, email, phone"""
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        # 创建用户实例
        user = UserInfo(name=name,age=age,email=email,phone=phone)
        session.add(user) #添加到会话
        session.commit() #提交事务
        return {"messages": [f"数据已成功存储至Mysql数据库。"]}
    except Exception as e:
        session.rollback()  # 出错时回滚
        return {"messages": [f"数据存储失败，错误原因：{e}"]}
    finally:
        session.close()  # 关闭会话