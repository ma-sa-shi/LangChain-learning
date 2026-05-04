import os
from utils import reciprocal_rank_fusion
from prompts import query_gen_prompt, answer_prompt
from schema import MultiQuery
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_cohere import CohereRerank

model = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME"))
cohere_reranker = CohereRerank(model=os.getenv("COHERE_MODEL_NAME", "rerank-multilingual-v3.0"), top_n=4)
structed_llm = model.with_structured_output(MultiQuery)

query_generation_chain = query_gen_prompt | structed_llm | (lambda x: x.queries)

# インスタンス化するためにchromaの起動が必要なretrieverを待つために、create_rag_chain関数を定義して、retrieverを引数として受け取る。
def create_rag_chain(retriever):
    multi_query_rag_chain = {
        "question": RunnablePassthrough(),
        # retriever.map()はChromaが並列で検索し文書のリストのリストを返す
        "context": (
            query_generation_chain # 質問を3~5のクエリに変換
            | retriever.map() # 各5件ずつ検索(計15~25件)
            | reciprocal_rank_fusion # RRFでrerankし上位20件に絞る
            | cohere_reranker # Cohereで上位4件に絞る
        )
    } | answer_prompt | model | StrOutputParser()
    return multi_query_rag_chain