from share.schema import GraphState
from share.chains import (
    query_generation_chain,
    answer_chain,
    cohere_reranker,
    grade_chain,
    failure_analysis_chain,
)
from share.vectorstore import get_retriever
from share.utils import reciprocal_rank_fusion


def generate_queries_node(state: GraphState):
    """質問を複数のクエリに変換するノード。
    Args:
        state (GraphState): 現在のグラフの状態
    Returns:
        dict: 生成されたクエリを含む辞書 {"queries": list[str]}
    """
    feedback = (
        state.get("feedback")[-1]
        if state.get("feedback")
        else "これは最初の試行です。最適な検索クエリを生成してください。"
    )
    queries = query_generation_chain.invoke(
        {"question": state.get("question"), "feedback": feedback}
    )
    return {"queries": [queries]}


def retrieve_node(state: GraphState):
    """クエリを基に情報検索して回答生成のためのdocumentsを作成するノード。
    Args:
        state (GraphState): 現在のグラフの状態
    Returns:
        dict: 検索されたドキュメントを含む辞書 {"documents": list[Document]}
    """
    retriever = get_retriever()
    raw_docs = retriever.map().invoke(state.get("queries")[-1])
    fused_docs = reciprocal_rank_fusion(raw_docs)
    final_docs = cohere_reranker.compress_documents(fused_docs, state.get("question"))
    return {"documents": [final_docs]}


# documentsを基に回答を生成するノード
def generate_answer_node(state: GraphState):
    """documentsを基に回答を生成するノード。
    Args:
        state (GraphState): 現在のグラフの状態
    Returns:
        dict: 生成された回答を含む辞書 {"answer": str}
    """
    response = answer_chain.invoke(
        {"question": state.get("question"), "context": state.get("documents")[-1]}
    )
    return {"answer": response}


def grade_answer_node(state: GraphState):
    """生成された回答を評価するノード。
    Args:
        state (GraphState): 現在のグラフの状態
    Returns:
        dict: 評価結果と評価理由を含む辞書 {"evaluation": str, "feedback": str, "loop_step": int}
    """
    result = grade_chain.invoke(
        {
            "question": state.get("question"),
            "answer": state.get("answer"),
            "context": state.get("documents")[-1],
        }
    )
    return {
        "evaluation": [result.evaluation],
        "feedback": [result.feedback],
        "loop_step": state.get("loop_step", 0) + 1,
    }


def failure_analysis_node(state: GraphState):
    """複数回の回答生成を行っても十分な回答が得られない場合に、原因分析するノード。
    Args:
        state (GraphState): 現在のグラフの状態
    Returns:
        dict: 分析結果を含む辞書 {"failure_analysis": str}
    """
    analysis = failure_analysis_chain.invoke(
        {
            "question": state.get("question"),
            "initial_queries": state.get("queries")[0],
            "initial_context": state.get("documents")[0],
            "initial_feedback": state.get("feedback")[0],
            "retry_feedback": state.get("feedback")[-1],
        }
    )
    return {"failure_analysis": analysis}
