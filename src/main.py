from dotenv import load_dotenv
from chains import create_rag_chain
from vectorstore import get_retriever

load_dotenv()

rag_chain = create_rag_chain(get_retriever())

question = ("質問を入力してください: ")
rag_chain.invoke({"question": question})