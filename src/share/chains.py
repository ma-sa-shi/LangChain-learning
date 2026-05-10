"""ワークフローで使用するチェーンを定義するモジュール。
LLMを使用して、質問から複数の検索クエリを生成し、ベクトルストアからドキュメントを検索し、回答を生成し、回答を評価するチェーンを定義する。
"""
import os
from share.utils import reciprocal_rank_fusion
from share.prompts import query_gen_prompt, answer_prompt, grade_prompt, failure_analysis_prompt
from share.schema import MultiQuery, GradeAnswer
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereRerank

model = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME", "gpt-5-nano"))
cohere_reranker = CohereRerank(model=os.getenv("COHERE_MODEL_NAME", "rerank-v3.5"), top_n=4)

query_generation_chain = query_gen_prompt | model.with_structured_output(MultiQuery) | (lambda x: x.queries)
answer_chain = answer_prompt | model | StrOutputParser()
grade_chain = grade_prompt | model.with_structured_output(GradeAnswer)
failure_analysis_chain = failure_analysis_prompt | model | StrOutputParser()

# インスタンス化するためにchromaの起動が必要なretrieverを待つために、create_rag_chain関数を定義して、retrieverを引数として受け取る。
def create_rag_chain(retriever):
    """RAGチェーンを作成する関数。
    Args:
        retriever: ベクトルストアからドキュメントを検索するためのリトリーバー
    Returns:
        multi_query_rag_chain: RAGチェーン
    """
    def rerank_docs(inputs):
        return cohere_reranker.compress_documents(
            documents=inputs.get("context"),
            query=inputs.get("question")
        )
    multi_query_rag_chain = {
        "question": RunnablePassthrough(),
        # retriever.map()はChromaが並列で検索し文書のリストのリストを返す
        "context": (
            query_generation_chain   # 質問を5個のクエリに変換
            | retriever.map()        # 各5件ずつ検索(計15~25件)
            | reciprocal_rank_fusion # RRFでrerankし上位20件に絞る
        )
    } | RunnablePassthrough.assign(
        context=RunnableLambda(rerank_docs) # cohereでrerankし上位4件に絞る
    ) | answer_chain
    return multi_query_rag_chain