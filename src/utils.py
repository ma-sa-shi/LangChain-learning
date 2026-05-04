from langchain_core.documents import Document


def reciprocal_rank_fusion(
        retriever_outputs: list[list[Document]],
        k: int= 60,
        top_n: int = 20
) -> list[Document]:
    # {スコア, ドキュメントの内容}
    doc_score_map = {}

    for docs in retriever_outputs:
        for rank, doc in enumerate(docs):
            content = doc.page_content
            if content not in doc_score_map:
                doc_score_map[content] = [0.0, doc]
            doc_score_map[content][0] += 1 / (rank + k)

    sorted_items = sorted(
        doc_score_map.items(),
        key=lambda x: x[1][0],
        reverse=True # 降順
    )

    reranked_docs = [doc for _, (score, doc) in sorted_items[:top_n]]

    return reranked_docs