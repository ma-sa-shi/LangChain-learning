
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_file(file_paths: list[str | Path]) -> list[Document]:
    all_docs = []
    for path_str in file_paths:
        path = Path(path_str)
        if not path.exists():
            print(f"警告: ファイルが存在しません: {path}")
            continue
    ext = path.suffix.lower()
    if ext in [".txt", ".md", ".rst"]:
        loader = TextLoader(str(path), encoding="utf-8")
    elif ext == ".pdf":
        loader = PyPDFLoader(str(path))
    else:
        print(f"未対応の形式です: {ext}")
    all_docs.extend(loader.load())
    return all_docs

def split_documents(documents: list[Document]) -> list[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators = [
            "\n\n.. class::",      # クラス定義
            "\n\n.. method::",     # メソッド定義
            "\n\n.. attribute::",  # 属性定義
            "\n\n.. function::",   # モジュールレベルの関数
            "\n\n",                # 段落
            "\n",                  # 行
            " "                    # 単語
        ]
    )
    return text_splitter.split_documents(documents)

def reciprocal_rank_fusion(
        retriever_outputs: list[list[Document]],
        k: int= 60,
        top_n: int = 20
) -> list[Document]:
    # {スコア, ドキュメントの内容}
    doc_score_map = {}

    for docs in retriever_outputs:
        for rank, doc in enumerate(docs):
            content = doc.page_content
            if content not in doc_score_map:
                doc_score_map[content] = [0.0, doc]
            doc_score_map[content][0] += 1 / (rank + k)

    sorted_items = sorted(
        doc_score_map.items(),
        key=lambda x: x[1][0],
        reverse=True # 降順
    )

    reranked_docs = [doc for _, (score, doc) in sorted_items[:top_n]]

    return reranked_docs