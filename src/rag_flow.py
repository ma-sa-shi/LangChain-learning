import os
import argparse
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

def save_to_chromadb(file_path, embedding_model, chunk_size=500, chunk_overlap=50, db_path="./chroma_db"):

    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(model=embedding_model)

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=db_path
    )

    print(f"保存完了: {len(docs)}個のチャンクが'{db_path}'に保存されました。")
    return vectorstore


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ファイルをベクトル化してChromaDBに保存する")
    parser.add_argument("--file_path", type=str, default="/langchain_learning/docs/*.txt", help="取込むドキュメントのファイルパス")
    parser.add_argument("--chunk_size", type=int, default=500, help="チャンクサイズ")
    parser.add_argument("--chunk_overlap", type=int, default=50, help="チャンクオーバーラップ")
    parser.add_argument("--db_path", type=str, default="/langchain_learning/chroma_db", help="DB保存先パス")
    args = parser.parse_args()

    file_path = args.file_path
    chunk_size = args.chunk_size
    chunk_overlap = args.chunk_overlap

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL_NAME", "text-embedding-3-small")

    save_to_chromadb(
        file_path=args.file_path,
        embedding_model=embedding_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        db_path=args.db_path
    )



