from langchain_core.prompts import ChatPromptTemplate # 聊天模板
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

llm =ChatOpenAI(
    model = 'deepseek-chat',
    base_url = 'https://api.deepseek.com/v1',
    api_key='sk-40aac449e559480b9676b129ff8a20dc',
)

# 根据message生成提示词模板
prompt = ChatPromptTemplate.from_messages([
   ("system","你是世界级的技术专家"), # 定义角色
   ("user","{input}")
])

# 创建一个字符串输出解析器
output_parser = StrOutputParser()

# 通过langchain的链式调用，生成一个chain
chain = prompt | llm | output_parser

result = chain.invoke({"input": "请你帮我写一篇关于AI的技术文章，100字"})
print(result)
# content='AI技术正在重塑全球产业格局。作为世界级专家，我认为当前最关键的突破在于多模态大模型的发展。这些模型不仅能处理文本，还能理解图像、视频和音频，实现了真正的跨模态学习。
# Transformer架构的持续优化使模型参数量突破万亿级别，而稀疏激活技术则大幅降低了计算成本。特别值得注意的是，AI系统开始展现初步的推理能力和世界知识，这为通用人工智能(AGI)奠定了基础。
# 未来3-5年，我们或将见证AI在科学发现领域的革命性应用。'
# additional_kwargs={'refusal': None} response_metadata={'token_usage': {'completion_tokens': 110, 'prompt_tokens': 19, 'total_tokens': 129, 'completion_tokens_details': None, 'prompt_tokens_details': {'audio_tokens': None, 'cached_tokens': 0}, 'prompt_cache_hit_tokens': 0, 'prompt_cache_miss_tokens': 19}, 'model_name': 'deepseek-chat', 'system_fingerprint': 'fp_8802369eaa_prod0623_fp8_kvcache', 'id': 'ebeb4671-d91d-4b87-b935-2dafa36cf2f2', 'service_tier': None, 'finish_reason': 'stop', 'logprobs': None} id='run--20996ab8-4b2f-45be-a4ee-ddb08cddab55-0' usage_metadata={'input_tokens': 19, 'output_tokens': 110, 'total_tokens': 129, 'input_token_details': {'cache_read': 0}, 'output_token_details': {}}

# 添加输出解析器后
# 人工智能（AI）正以惊人的速度改变世界。从深度学习到自然语言处理，AI技术已渗透至医疗、金融、制造等领域，提升效率并创造新机遇。大模型如GPT-4展现了强大的生成能力，而强化学习在自动驾驶等领域取得突破。然而，AI的伦理与安全问题仍需重视。未来，随着算法优化与算力提升，AI将继续推动社会进步，成为人类创新的核心驱动力。