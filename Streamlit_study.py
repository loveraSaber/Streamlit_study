from langchain.agents import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from datetime import datetime
from langchain_core.tools import  tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor
load_dotenv(r"D:\Project\Python_Project\AILLM\RAG\config\.env")
os.environ['HTTP_PROXY'] = ''
os.environ['HTTPS_PROXY'] = ''
os.environ['NO_PROXY'] = 'siliconflow.cn,localhost,127.0.0.1'
api_key=os.getenv("api_key")
url_chat= os.getenv("chat_url")
embedding_url= os.getenv("embedding_url")
chat_model= os.getenv("chat_model")
embedding_model= os.getenv("embedding_model")
system_prompt="""
你是一个助手，请根绝用户的问题，给出专业的回答
"""
prompt="""用户的问题:{question}
"""
prompt_template=ChatPromptTemplate([
    ("system",system_prompt),
    ("user",prompt)
])
llm = ChatOpenAI(
    api_key=api_key,
    base_url=url_chat,
    model=chat_model,
    temperature=0.7,
)
embeddings=OpenAIEmbeddings(
    api_key=api_key,
    base_url=embedding_url,
    model=embedding_model,
)
@tool
def get_current_time(city:str="北京")->str:
    """获取当前时间"""
    return f"{city} 当前时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
prompt = PromptTemplate.from_template("""
    你是一个可以调用工具的智能助手。

    你可以使用以下工具：
    {tools}

    请严格按照以下格式思考和回答：

    Question: {input}
    Thought: 是否需要调用工具
    Action: 选择一个工具，必须是 [{tool_names}] 中的一个
    Action Input: 工具参数
    Observation: 工具返回结果
    Thought: 我已经知道答案了
    Final Answer: 给用户的最终答案

    {agent_scratchpad}
    """)
agent=create_react_agent(llm,tools=[get_current_time],prompt=prompt)
executor=AgentExecutor(agent=agent,tools=[get_current_time],verbose=True)
# result = executor.invoke({"input": "现在是什么时间？"})
# print(result["output"])

import streamlit as st

st.title("🤖 Agent 对话系统")

# 初始化对话历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 输入框
query = st.chat_input("请输入你的问题")

if query:
    # 显示用户消息
    st.session_state.messages.append(
        {"role": "user", "content": query}
    )
    with st.chat_message("user"):
        st.markdown(query)

    # 调用 Agent
    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            result = executor.invoke({"input": query})
            answer = result["output"]
            st.markdown(answer)

    # 保存助手回复
    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
