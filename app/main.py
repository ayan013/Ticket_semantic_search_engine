import json
from retriever import SemanticRetrival
from embedding_service import EmbeddingService
from evaluator import RetrievalEvaluator


with open("data/tickets.json","r") as file:
    tickets = json.load(file)

with open("data/eval_set.json","r") as file:
    queries = json.load(file)

texts = [ticket["text"] for ticket in tickets]


embedding_service = EmbeddingService() # Embedding object creation
document_embeddings = embedding_service.embedded_documents(texts) #document embedding

retriever = SemanticRetrival(tickets,document_embeddings) # semantic retrieval object creation
evaluator = RetrievalEvaluator(retriever,embedding_service)

#for top_k in [1,3,5]:
result = evaluator.recall_at_k(query = "I returned the purchase but still don't have my money",relevant_id = [9, 11],top_k = 3)
print(f"{round(result,2)}")


