"""ワークフローで使用するプロンプトを定義するモジュール。
LLMを使用して、質問から複数の検索クエリを生成し、ベクトルストアからドキュメントを検索し、回答を生成し、回答を評価するためのプロンプトを定義する。
"""

from langchain_core.prompts import ChatPromptTemplate

# クエリ生成用プロンプト
query_gen_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "ユーザーの質問に対して、検索精度を高めるための3~5個のクエリを生成してください。"
            "前回の試行でフィードバックがある場合は、それを踏まえて異なるアプローチのクエリを考えてください。",
        ),
        ("human", "質問: {question}\nフィードバック: {feedback}"),
    ]
)

# 回答用プロンプト
answer_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "ユーザーの質問に対して、以下のコンテキストだけを踏まえて回答してください。\n\nコンテキスト: {context}",
        ),
        ("human", "質問: {question}"),
    ]
)

# 評価用プロンプト
grade_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "コンテキストを基に以下の質問と回答を評価してください。評価はuseful、useless、hallucinationのいずれかで、理由も述べてください。\n\nコンテキスト: {context}",
        ),
        ("human", "質問: {question}\n\n回答: {answer}"),
    ]
)

# 失敗分析用プロンプト
failure_analysis_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
     複数回の回答生成を行いましたが、ユーザーの質問に対して十分な回答が得られませんでした。

     以下のいずれのケースに該当するか、あるいは他に原因があるか検討してください。
     ・ベクトルストア内に必要な情報がそもそも存在しない\n
     ・ユーザーの質問が曖昧すぎて、絞り込みができなかった\n
     ・初回のクエリが適切ではなく、必要な情報を引き出せなかった\n
     ・その他（具体的に記述してください）
    """,
        ),
        (
            "human",
            """
     分析データを基に、適切な回答を生成できなかった原因を教えてください。
    ### 分析データ
     1. 質問: {question}\n
     2. 初回のクエリ: {initial_queries}\n
     3. 初回のコンテキスト: {initial_context}\n
     4. 初回の回答へのフィードバック: {initial_feedback}\n
     5. 再試行後の回答へのフィードバック: {retry_feedback}
    """,
        ),
    ]
)
