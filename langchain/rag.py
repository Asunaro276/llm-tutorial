from langchain_community.document_loaders import PyPDFLoader
from langchain_deepseek import ChatDeepSeek
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain.schema import SystemMessage
from langchain_core.output_parsers import StrOutputParser
from operator import itemgetter
import requests
from dotenv import load_dotenv
load_dotenv()

url = "https://docs.aws.amazon.com/ja_jp/amazondynamodb/latest/developerguide/dynamodb-dg.pdf"
filename = "dynamodb.pdf"

with open(filename, "wb") as file:
    file.write(requests.get(url).content)

loader = PyPDFLoader(filename)
pages = loader.load()
print(f"ページ数: {len(pages)}")

n = 110
print(f"{n}ページ目: {pages[n].page_content[:500]}")

python_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
splits = python_splitter.split_documents(pages)
print(f"分割数: {len(splits)}")
print(f"{n}チャンク目: {splits[n].page_content[:500]}")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

content = splits[n].page_content
vector = embeddings.embed_query(content)
print(f"ベクトルの次元数: {len(vector)}")
print(f"ベクトルの最初の10要素: {vector[:10]}")

vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="chroma_db"
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt_template = ChatPromptTemplate.from_messages([
    SystemMessage("あなたは有能なアシスタントです。"),
    MessagesPlaceholder("chat_history"),
    HumanMessagePromptTemplate.from_template(
        "与えられた文脈に基づいて、次の質問に答えてください。\n: {context}\n質問: {question}"
    ),
])

llm = ChatDeepSeek(model="deepseek-chat", temperature=0)

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt_template
    | llm
    | StrOutputParser()
)

history = []
answer = rag_chain.invoke({
    "question": "Dyanmodbのモデリングのコツ",
    "chat_history": history,
})
print(answer)
