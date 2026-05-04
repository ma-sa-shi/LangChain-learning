import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


def get_retriever(k: int = 4):
    PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    return  vectorstore.as_retriever(search_kwargs={"k": k})

def add_documents_to_db(texts: list[str]):
    PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    docs = [Document(page_content=t) for t in texts]

    Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=PERSIST_DIRECTORY
    )
    print(f"{len(docs)}件のドキュメントを追加しました。")