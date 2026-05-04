from pydantic import BaseModel, Field

class MultiQuery(BaseModel):
    queries: list[str] = Field(...,min_items=3, max_items=5, description="LLMが生成する検索クエリ") # ...はEllipsisと言い必須であることを表す
