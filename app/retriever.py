import numpy as np

class SemanticRetrival:

    def __init__(self,tickets: list[dict], documents_embedding: np.ndarray):
        self.tickets = tickets
        self.documents_embedding = documents_embedding

    def cosine_similarity(self,query_embedding: np.ndarray) -> np.ndarray:
        dot_product = np.dot(self.documents_embedding,query_embedding)
        document_magnitude = np.linalg.norm(self.documents_embedding, axis=1)
        query_magnitude = np.linalg.norm(query_embedding)

        similarity = dot_product / (document_magnitude * query_magnitude)

        return similarity

    def search(self, similarity_score: np.ndarray) -> list[dict]:
            result = []
            #top_k = len(self.tickets)
            docs = [document for document in self.tickets if document["id"] == 5]
            for document, score in zip(docs,similarity_score):

                    result.append({
                    "id":document["id"],
                    "category": document["category"],
                    "text": document["text"],
                    "score": round(float(score),4)
                    })

            #result.sort(key = lambda item:item["score"], reverse= True)
            return result




