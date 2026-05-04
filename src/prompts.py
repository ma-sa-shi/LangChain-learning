from langchain_core.prompts import ChatPromptTemplate

# クエリ生成用プロンプト
query_gen_prompt = ChatPromptTemplate.from_messages([
    ("system", "ユーザーの質問に対して、検索精度を高めるための3〜5つのクエリを生成してください。"),
    ("human", "{question}")
])

# 回答用プロンプト
answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "ユーザーの質問に対して、以下の文脈だけを踏まえて回答してください。\n\n{context}"),
    ("human", "{question}")
])