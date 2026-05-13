from dotenv import load_dotenv
from graph.workflow import compiled

load_dotenv()

question = input("質問を入力してください: ")
initial_state = {"question": question}
result = compiled.invoke(initial_state)

if result.get("failure_analysis"):
    print("\n回答生成に失敗しました。\n原因分析")
    print(result.get("failure_analysis"))

print("\n回答")
print(result.get("answer"))
