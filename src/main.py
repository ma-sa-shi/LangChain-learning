from dotenv import load_dotenv
from share.chains import create_rag_chain
from share.vectorstore import get_retriever
from graph.workflow import compiled

load_dotenv()

rag_chain = create_rag_chain(get_retriever())

question = input("質問を入力してください: ")
initial_state = {"question": question}
result = compiled.invoke(initial_state)
print(result.get("answer"))