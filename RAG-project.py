# 初始化apikey   模型选用智谱清言ai
import os
os.environ["ZHIPUAI_API_KEY"] = ""
# 获取 serapi key
os.environ["SERPAPI_API_KEY"] = ""
# WebBaseLoader 的user agent
os.environ["USER_AGENT"] = ""


# 导包，构建模型
from langchain_community.chat_models import ChatZhipuAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import MessagesPlaceholder, ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.agent_toolkits.load_tools import load_tools
from langchain.agents import (initialize_agent, AgentType)
from langchain.chains import LLMMathChain
from langchain.agents import Tool
from langchain.chains import LLMChain
import re
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader

# 模型选择智谱清言
zhipuai_chat_model = ChatZhipuAI()
chat_model = zhipuai_chat_model

# 加载网页数据
loader = WebBaseLoader(
    web_path="https://baike.baidu.com/item/%E5%94%90%E5%B1%B1%E5%B8%82/8404217"
)
docs = loader.load()
# 将docs索引到向量存储
EMBEDDING_DEVICE = "cpu"
embeddings = HuggingFaceEmbeddings(model_name="..\models\m3e-base", model_kwargs={'device': EMBEDDING_DEVICE})


# 生成词切分器
text_splitter = RecursiveCharacterTextSplitter()
# 对 load 进来的文档 进行分词&切分
documents = text_splitter.split_documents(documents=docs)
# 建立索引：将词向量存储向量数据库
vector = FAISS.from_documents(documents=documents, embedding=embeddings)

# 使用文本信息作为大模型数据源
base_dir = "..\mydocuments"
txt_documents = []
# 开始遍历指定文件夹
for filename in os.listdir(base_dir):
    # 构建完成的文件名（含有路径信息）
    file_path = os.path.join(base_dir, filename)
    # 分别使用不同的加载器加载各类不同的文档
    if filename.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        txt_documents.extend(loader.load())
    elif filename.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
        txt_documents.extend(loader.load())
    elif filename.endswith(".txt"):
        loader = TextLoader(file_path,encoding = "utf-8")
        txt_documents.extend(loader.load())

txt_chunked_documents = text_splitter.split_documents(documents=txt_documents)
# 切换向量数据库模型 Qdrant
txt_vector = Qdrant.from_documents(
    documents=txt_chunked_documents,
    embedding=embeddings,
    location=":memory:",
    collection_name="my_documents",
)

# 使用网页资源的检索器
retriever = vector.as_retriever()

# 使用文件资源的检索器
txt_retriever = txt_vector.as_retriever()

# 生成 ChatModel 会话的提示词
prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
     ("user", "{input}"),
     ("system", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
])
# 生成含有历史信息的检索链
retriever_chain = create_history_aware_retriever(chat_model, retriever, prompt)
txt_retriever_chain = create_history_aware_retriever(chat_model, txt_retriever, prompt)
# 继续对话，记住检索到的文档等信息
prompt = ChatPromptTemplate.from_messages([
    ("system", """
    你会优先使用tools，如果检索不到信息再使用你的知识库。\n\n
    你会优先使用retrieval_chain工具回答问题，如果在retrieval_chain检索不到需要的信息，你可以使用Search工具。\n\n
    Answer the user's questions based on the below context:\n\n
    {context}\n\n
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}"),
])
document_chain = create_stuff_documents_chain(chat_model, prompt)
retrieval_chain = create_retrieval_chain(retriever_chain, document_chain)
txt_retrieval_chain = create_retrieval_chain(txt_retriever_chain, document_chain)
# 模拟一个历史会话记录
chat_history = []
# 加载tools
tools = load_tools(tool_names=["serpapi","llm-math"], llm=chat_model)
#定义retrieval_tool函数，将其导入进agent
def retrieval_tool(query):
    return retrieval_chain.invoke({
        "chat_history": chat_history,
        "input":query
    })["answer"]

# 实例为 agent 工具
retrieval_tool_instance = Tool(
    name="retrieval_tool",
    description="Take this in priority status,"
                 "This tool handles web page retrieval and "
                 "questionn answering based on context history.",
    func=retrieval_tool,
)

# 将 retriever链作为tool加载到agent
tools.append(retrieval_tool_instance)
# 下述同理
def txt_retrieval_tool(query):
    return txt_retrieval_chain.invoke({
        "chat_history": chat_history,
        "input":query
    })["answer"]

# 实例为 agent 工具
txt_retrieval_tool_instance = Tool(
    name="txt_retrieval_tool",
    description="This tool handles documents retrieval and "
                 "questionn answering based on context history.",
    func=txt_retrieval_tool,
)

# 将 txt_retriever链作为tool加载到agent
tools.append(txt_retrieval_tool_instance)

#  Agent初始化
agent = initialize_agent(
    tools=tools,
    llm=chat_model,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    handle_parsing_errors=True,
)

print("开始与大模型对话")
while True:
    human_message = input("请输入问题（输入 '结束' 结束）：")
    if human_message == "结束":
        break
    response = agent.invoke({
                "input": human_message,
                "chat_history": chat_history,
            })
    ai_message = response["output"]
    print("回答：", ai_message)
    chat_history.append(HumanMessage(content=human_message))
    chat_history.append(AIMessage(content=ai_message))

print("对话已结束")

