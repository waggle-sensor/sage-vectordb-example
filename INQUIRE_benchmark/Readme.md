IN PROGRESS

This uses the same setup as [Hybrid Search](../HybridSearch_example/) so that we can benchmark [Hybrid Search](../HybridSearch_example/) using [INQUIRE](https://github.com/inquire-benchmark/INQUIRE).

## Usage

This benchmark is supposed to be used in conjuction with [Hybrid Search](../HybridSearch_example/). The Makefile references components that are deployed in [Hybrid Search](../HybridSearch_example/). The Makefile in here deploys additional containers that are used to run the INQUIRE Benchmark.

## References
- [Weaviate Blog: NDCG](https://weaviate.io/blog/retrieval-evaluation-metrics#normalized-discounted-cumulative-gain-ndcg)
- [RAG Evaluation](https://weaviate.io/blog/rag-evaluation)
- [Scikit-Learn NDCG](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)
- [A Guide on NDCG](https://www.aporia.com/learn/a-practical-guide-to-normalized-discounted-cumulative-gain-ndcg/)
- [Weaviate: Batch import](https://weaviate.io/developers/weaviate/manage-data/import)
- [Weaviate: Imports in Detail](https://weaviate.io/developers/weaviate/tutorials/import#data-import---best-practices)