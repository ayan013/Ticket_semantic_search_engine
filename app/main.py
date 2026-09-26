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

for hit in [1,3,5]:
    for query in queries:
        if query["id"] == "q004":
            exact_query = query["query"]
            relevant_doc = query["relevant_ticket_ids"]
    result = evaluator.hit_at_k(exact_query,relevant_doc,hit)
    print(f"Top: {hit}:result: {result}")


