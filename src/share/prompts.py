"""ワークフローで使用するプロンプトを定義するモジュール。
LLMを使用して、質問から複数の検索クエリを生成し、ベクトルストアからドキュメントを検索し、回答を生成し、回答を評価するためのプロンプトを定義する。
"""
from langchain_core.prompts import ChatPromptTemplate

# クエリ生成用プロンプト
query_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", "ユーザーの質問に対して、検索精度を高めるための3〜5つのクエリを生成してください。"
    "前回の試行でフィードバックがある場合は、それを踏まえて異なるアプローチのクエリを考えてください。"),
    ("human", "質問: {question}\nフィードバック: {feedback}")
])

# 回答用プロンプト
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "ユーザーの質問に対して、以下のコンテキストだけを踏まえて回答してください。\n\nコンテキスト: {context}"),
    ("human", "質問: {question}")
])

# 評価用プロンプト
grade_prompt = ChatPromptTemplate.from_messages([
    ("system", "コンテキストを基に以下の質問と回答を評価してください。評価はuseful、useless、hallucinationのいずれかで、理由も述べてください。\n\nコンテキスト: {context}\n\n質問: {question}\n\n回答: {answer}"),
    ("human", "質問: {question}\n\n回答: {answer}")
])