import os
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

load_dotenv()
topic = input("トピックを入力してください: ")
prompt = ChatPromptTemplate.from_template("{topic}について、簡潔に1行で説明してください。")
model = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME"))
parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({topic})
print(result)