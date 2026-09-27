

class RetrievalEvaluator:

    def __init__(self,retriever, embedding_service):
        self.retriever = retriever
        self.embedding_service = embedding_service

    def hit_at_k(self,query: str,relevant_id: list, top_k: int) -> int:

        embedded_query = self.embedding_service.embedded_query(query)

        similarity_scores = self.retriever.cosine_similarity(embedded_query)

        result = self.retriever.search(similarity_scores,top_k=top_k)
        for ticket in result:
            if ticket["id"] in relevant_id:
                return 1
        return 0

    def evaluate_hit_at_k(self,evaluation_queries: list[dict],top_k: int) -> float:
        hit = 0

        for query in evaluation_queries:
            result = self.hit_at_k(query["query"],query["relevant_ticket_ids"],top_k)
            hit = hit+result

        return hit/len(evaluation_queries)

    def recall_at_k(self,query,relevant_id: list, top_k) -> float:

        embedded_query = self.embedding_service.embedded_query(query)

        similarity_scores = self.retriever.cosine_similarity(embedded_query)
        result = self.retriever.search(similarity_scores, top_k=top_k)
        relevant_ids = set(relevant_id)
        result_ids = {ticket["id"] for ticket in result}
        intersection = relevant_ids & result_ids
        recall = len(intersection)/len(relevant_ids)
        return recall



