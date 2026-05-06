"""ワークフローで使用するスキーマを定義するモジュール。
GraphStateはワークフローの状態を表すクラスで、質問、検索クエリ、検索されたドキュメント、生成された回答、評価結果、評価理由、ループ回数などの属性を持つ。
MultiQueryはLLMが生成する複数の検索クエリを表すクラスで、queries属性を持つ。
GradeAnswerはLLMが生成する回答の評価を表すクラスで、evaluation属性とfeedback属性を持つ。"""
# バリデーションが必要なクラス属性はBaseModelを継承して定義し、それ以外のクラス属性はTypedDictを継承して定義する。
# 追加が必要なクラス属性はTypedDictを継承して定義する。
from typing import Annotated, Literal, TypedDict
import operator
from pydantic import BaseModel, Field
from langchain_core.documents import Document

class MultiQuery(BaseModel):
    """LLMが生成する複数の検索クエリを表すクラス。
    Attributes:
        queries (list[str]): LLMが生成する検索クエリのリスト。3~5のクエリが必要。
    """
    queries: list[str] = Field(
        ...,
        min_items=3,
        max_items=5,
        description="LLMが生成する検索クエリ"
    ) # ...はEllipsisと言い必須であることを表す

class GradeAnswer(BaseModel):
    """LLMが生成する回答の評価を表すクラス。
    Attributes:
        evaluation (Literal["useful", "useless", "hallucination"]): 質問に対する回答の評価。"useful"、"useless"、"hallucination"のいずれかでなければならない。
        feedback (str): 評価理由。自由記述のテキスト。
    """
    evaluation: Literal["useful", "useless", "hallucination"] = Field(
        ...,
        description="質問に対する回答の評価"
    )
    feedback: str = Field(..., description="評価理由")

class GraphState(TypedDict):
    """ワークフローの状態を表すクラス。
    Attributes:
    question (str): ユーザーからの質問
    queries (list[str]): LLMが生成する検索クエリのリスト
    documents (list[Document]): ベクトルストアから検索されたドキュメントのリスト
    answer (str): LLMが生成する回答
    evaluation (str): LLMが生成する回答の評価
    feedback (str): LLMが生成する回答の評価理由
    loop_step (int): ワークフローのループ回数。初回は0で、再試行するたびに1ずつ増加する。
    """
    question: str
    queries: list[str]
    documents: Annotated[list[Document], operator.add]
    answer: str
    evaluation: str
    feedback: str
    loop_step: int