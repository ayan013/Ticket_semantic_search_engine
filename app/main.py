import json
from retriever import SemanticRetrival
from embedding_service import EmbeddingService


with open("data/tickets.json","r") as file:
    tickets = json.load(file)


texts = [ticket["text"] for ticket in tickets if ticket["id"] == 5]
print(texts)
queries = ["Money left my bank account but the purchase failed",
           "Payment was deducted but the purchase failed",
           "Money was deducted but my order was not created",
           "My payment succeeded but no order was created"
           ]

embedded_service = EmbeddingService() # Embedding object creation
document_embeddings = embedded_service.embedded_documents(texts) #document embedding

for query in queries:
    query_embeddings = embedded_service.embedded_query(query) # each query embedding


    retriever = SemanticRetrival(tickets,document_embeddings) # semantic retrieval object creation
    similarity_score = retriever.cosine_similarity(query_embeddings) #Cosine similarity
    result = retriever.search(similarity_score) # Search result -> Top k
    print(f"User query - {query},\n {result}")








