

class RetrievalEvaluator:

    def __init__(self,retriever, embedding_service):
        self.retriever = retriever
        self.embedding_service = embedding_service

    def hit_at_k(self,query,relevant_id: list, top_k) -> int:

        embedded_query = self.embedding_service.embedded_query(query)

        similarity_scores = self.retriever.cosine_similarity(embedded_query)

        result = self.retriever.search(similarity_scores,top_k=top_k)
        for ticket in result:
            if ticket["id"] in relevant_id:
                return 1
        return 0