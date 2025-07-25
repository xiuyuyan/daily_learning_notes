from pydantic import BaseModel,Field
from typing import Union, Optional

# 实时搜索模型
class SearchQuery(BaseModel):
    query:str = Field(description="Question for networking queries")

# 查询天气模型（mock）
class WeatherLoc(BaseModel):
    location:str = Field(description="Location of weather station")

# 操作数据库模型
class UserInfo(BaseModel):
    """Extracted user information, such as name, age, email, and phone number, if relevant."""
    name:str = Field(description="Name of user")
    age:Optional[int] = Field(description="Age of user")
    email:str = Field(description="Email of user")
    phone:Optional[str] = Field(description="Phone number of user")

# 正常生成模型回复的模型
class ConversationalResponse(BaseModel):
    """Respond to the user's query in a conversational manner. Be kind and helpful."""
    response:str = Field(description="A conversational response to the user's query")

# 最终响应的模型，可以是用户信息或一般响应
class FinalResponse(BaseModel):
    final_output:Union[ConversationalResponse,WeatherLoc,UserInfo,SearchQuery]