import json
from retriever import SemanticRetrival
from embedding_service import EmbeddingService
from evaluator import RetrievalEvaluator
from data_loader import load_json



tickets = load_json("data/tickets.json")
evaluation_queries = load_json("data/evaluation_queries.json")

texts = [ticket["text"] for ticket in tickets]


embedding_service = EmbeddingService() # Embedding object creation, This hold document and query embedding service
document_embeddings = embedding_service.embedded_documents(texts) #document embedding
retriever = SemanticRetrival(tickets,document_embeddings) # This object holds cosine similarity, search Top_k service.
evaluator = RetrievalEvaluator(retriever,embedding_service) # This object hold all embedding, similarity and top-k services.


for k in [1,3,5]:

    hit = evaluator.evaluate_hit_at_k(evaluation_queries,top_k=k)
    recall = evaluator.evaluate_recall_at_k(evaluation_queries,top_k=k)

    print(f"K={k} | Hit@K={hit:.2f} | Recall@K={recall:.2f}")

