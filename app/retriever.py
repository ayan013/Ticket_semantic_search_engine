# This file is used for similarity matching and searching documents based on top_k scoring
import numpy as np


class SemanticRetrival:

    def __init__(self,tickets: list[dict], documents_embedding: np.ndarray):
        self.tickets = tickets
        self.documents_embedding = documents_embedding

        document_norms = np.linalg.norm(documents_embedding,axis=1,keepdims=True)
        self.normalized_documents = (documents_embedding / document_norms)


    def cosine_similarity(self,query_embedding: np.ndarray) -> np.ndarray:
        dot_product = np.dot(self.documents_embedding,query_embedding)
        document_magnitude = np.linalg.norm(self.documents_embedding, axis=1)
        query_magnitude = np.linalg.norm(query_embedding)

        if query_magnitude == 0:
            raise ValueError("Query embedding has zero magnitude")
        if np.any(document_magnitude == 0):
            raise ValueError("One or more document embeddings have zero magnitude")
        
        similarity = dot_product / (document_magnitude * query_magnitude)

        return similarity

    def cosine_similarity_normalized(self,query_embedding:np.ndarray) -> np.ndarray:
        query_norm = np.linalg.norm(query_embedding)
        normalized_query = query_embedding/query_norm

        similarity_score = np.dot(self.normalized_documents,normalized_query)

        return similarity_score

    def search(self, similarity_score: np.ndarray,top_k) -> list[dict]:
            result = []
            for document, score in zip(self.tickets,similarity_score):
                    result.append({
                    "id":document["id"],
                    "category": document["category"],
                    "text": document["text"],
                    "score": round(float(score),4)
                    })

            result.sort(key = lambda item:item["score"], reverse= True)
            return result[:top_k]




