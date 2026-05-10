"""ワークフローの定義を行うモジュール。
StateGraphを使用して、LLMが複数の検索クエリを生成し、ベクトルストアからドキュメントを検索し、回答を生成して、回答を評価するワークフローを定義する。
"""

from langgraph.graph import StateGraph, END
from graph.nodes import generate_queries_node, retrieve_node, generate_answer_node, grade_answer_node, failure_analysis_node
from share.schema import GraphState

def decide_to_finish(state: GraphState):
    """評価結果に基づいて、ワークフローの次のステップを決定する関数。
    Args:
        state (GraphState): 現在のグラフの状態
    Returns:
        str: 次のステップを示す文字列 ("finish", "force_finish", "retry")
    """
    if state.get("evaluation")[-1] == "useful":
        return "finish"

    if state.get("loop_step") >= 1:
        return "force_finish"

    return "retry"

workflow = StateGraph(GraphState)

workflow.add_node("query_gen", generate_queries_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_node("grade_answer", grade_answer_node)
workflow.add_node("failure_analysis", failure_analysis_node)

workflow.set_entry_point("query_gen")
workflow.add_edge("query_gen", "retrieve")
workflow.add_edge("retrieve", "generate_answer")
workflow.add_edge("generate_answer", "grade_answer")

workflow.add_conditional_edges(
    "grade_answer",
    decide_to_finish,
    {
        "finish": END,
        "force_finish": "failure_analysis",
        "retry": "query_gen"
    }
)
workflow.add_edge("failure_analysis", END)

compiled = workflow.compile()