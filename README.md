# AI Ticket Semantic Search

Baseline semantic retrieval system using:

- Sentence Transformers
- NumPy
- cosine similarity
- Hit@K
- Recall@K

## Baseline

Embedding model:
all-MiniLM-L6-v2

Hit@1: 0.83
Hit@3: 0.92
Hit@5: 1.00

Recall@1: 0.67
Recall@3: 0.92
Recall@5: 1.00

## Known Failure

q004:
"Money left my bank account but the purchase failed"

Expected ticket:
5

Rank:
5