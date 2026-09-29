import json
from retriever import SemanticRetrival
from embedding_service import EmbeddingService
from evaluator import RetrievalEvaluator


with open("data/tickets.json","r") as file:
    tickets = json.load(file)

with open("data/eval_set.json","r") as file:
    queries = json.load(file)

texts = [ticket["text"] for ticket in tickets]


embedding_service = EmbeddingService() # Embedding object creation, This hold document and query embedding service
document_embeddings = embedding_service.embedded_documents(texts) #document embedding

retriever = SemanticRetrival(tickets,document_embeddings) # This object holds cosine similarity, search Top_k service.
evaluator = RetrievalEvaluator(retriever,embedding_service) # This object hold all embedding, similarity and top-k services.


# result = evaluator.evaluate_queries(queries,top_k=3)
# for query in result:
#     if query["hit"] == 0 or query["recall"] < 1.0:
#         print(query)

query = "Money left my bank account but the purchase failed"
embed_query = embedding_service.embedded_query(query)

#original cosine
score = retriever.cosine_similarity(embed_query)
results = retriever.search(score,top_k=5)
print("Original Cosine")
for result in results:
    print(result)


#normalized cosine
normalized_score = retriever.cosine_similarity_normalized(embed_query)
normalized_results = retriever.search(normalized_score,top_k=5)

print("Normalized cosine dot product")
for normalized_result in normalized_results:
    print(normalized_result)

