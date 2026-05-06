import os
from typing import Literal
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import  ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field

load_dotenv()

class Sentiment(BaseModel):
    sentiment: Literal["ポジティブ", "ネガティブ"] = Field(description="感情のラベル")

class Summary(BaseModel):
    summary: str = Field(description="文章の30文字以内の要約")

sentences = input("文章を入力してください: ")

model = ChatOpenAI(model=os.getenv("OPENAI_MODEL_NAME"))
promptA = ChatPromptTemplate.from_template("{sentences}について、ポジティブかネガティブか判定してください")
promptB = ChatPromptTemplate.from_template("{sentences}について30文字以内で要約してください")
# parser = StrOutputParser()

chain = (
    RunnableParallel({
        "sentiment": promptA | model.with_structured_output(Sentiment),
        "summary": promptB | model.with_structured_output(Summary)
    })
    | (lambda x: f"感情は{x.get('sentiment').sentiment}で、内容は{x.get('summary').summary}")
)

result = chain.invoke({"sentences": sentences})
print(result)