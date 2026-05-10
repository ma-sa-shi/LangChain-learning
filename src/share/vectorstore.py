"""
ベクトルストアの管理を行うモジュール。
Chromaを使用して、ドキュメントのベクトル化と検索を行う。
"""
import os
import sys
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from share.utils import load_file, split_documents

def add_files_to_db(file_paths: list[str]):
    """ファイルをベクトルストアに追加する関数。
    Args:
        file_paths (list[str]): 追加するファイルのパスのリスト
    """
    docs = split_documents(load_file(file_paths))
    if not docs:
        return
    PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-3-small"))
    Chroma(
        embedding_function=embeddings,
        persist_directory=PERSIST_DIRECTORY
    ).add_documents(docs)
    print(f"{len(docs)}件のドキュメントを追加しました。")

def get_retriever(k: int = 4):
    """ベクトルストアからドキュメントを検索するためのリトリーバーを取得する関数。
    Args:
        k (int): 検索するドキュメントの数
    Returns:
        VectorStoreRetriever: リトリーバー
    """
    PERSIST_DIRECTORY = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")

    embeddings = OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-3-small"))

    vectorstore = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings
    )
    return  vectorstore.as_retriever(search_kwargs={"k": k})

if __name__ == "__main__":
    # sys.argv[0]はスクリプト名、sys.argv[1:]はコマンドライン引数のリスト
    if len(sys.argv) > 1:
        target_files = sys.argv[1:]
        add_files_to_db(target_files)