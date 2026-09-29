#This file only used for embedding
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    def __init__(self):
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    def embedded_documents(self, documents: list[str]):
        embeddings = self.model.encode_document(documents, convert_to_numpy = True)
        return embeddings.astype("float32")

    def embedded_query(self, query: str):
        embeddings = self.model.encode_query(query, convert_to_numpy = True)
        return embeddings.astype("float32")




