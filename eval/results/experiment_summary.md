# VTU RAG Evaluation Results

| Config | Split | hit@1 | hit@3 | hit@5 | MRR | Refusal acc. | p50 latency | p95 latency |
|---|---|---|---|---|---|---|---|---|
| Baseline (BGE-small, 500 tokens) | dev | 0.69 | 0.88 | 0.88 | 0.78 | 85.0% | 0.030s | 0.083s |
| + chunk 300 / overlap 80 | dev | 0.69 | 0.88 | 0.88 | 0.78 | 85.0% | 0.021s | 0.041s |
| + MiniLM embeddings | dev | 0.31 | 0.69 | 0.94 | 0.54 | 20.0% | 0.026s | 0.311s |
| + reranker | dev | 0.69 | 0.81 | 0.88 | 0.76 | 85.0% | 0.202s | 0.530s |
| Final (BGE-small + Metadata Filter) | test | 0.50 | 0.83 | 0.88 | 0.68 | 86.7% | 0.021s | 0.033s |
