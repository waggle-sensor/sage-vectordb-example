# Sage w/Vector Database

A repo of examples of how Sage can be used with [Weaviate](https://github.com/weaviate/weaviate) and other VectorDBs for cool machine-learning related tasks.

## Examples

|Title|Language|Description|
|---|---|---|
| [Text/Image search using CLIP](/CLIP_example/) | Python  | Use text or images to search through Sage images using CLIP (multi2vec-clip Weaviate module).|
| [MultiModal search using ImageBind](/ImageBind_example/) | Python  | Use text,images,audio, & more to search through Sage images using ImageBind (multi2vec-bind Weaviate module).|
| [MultiModal Hybrid search using ImageBind and Gemma 3](/HybridSearch_example/) | Python  | Use text and images to search through Sage images using DFN Clip. Images are also captioned using Gemma 3 to conduct a Hybrid search where a keyword search and a vector search is conducted. WEAVIATE v4 IS USED HERE|
| [Sage Chat](/SageChat/) | Python  | This implements an AI agent that can use multiple functions including [MultiModal Hybrid search using ImageBind and Florence 2](/HybridSearch_example/). WEAVIATE v4 IS USED HERE|
| [INQUIRE Benchmark](/INQUIRE_benchmark/) | Python  | This scores our [MultiModal Hybrid search](/HybridSearch_example/) using [INQUIRE](https://github.com/inquire-benchmark/INQUIRE). WEAVIATE v4 IS USED HERE|
| TODO [Vectordb on the edge](https://www.jetson-ai-lab.com/tutorial_nanodb.html) | Python  | This example runs a vector database on the edge using nvidia's nanodb. |
| TODO [Monitoring Setup with Prometheus & Grafana](monitoring-prometheus-grafana) | yaml  | This example does not describe any use case, but rather shows a way of how to start, operate and configure Weaviate with Prometheus-Monitoring and a Grafana Instance with some sample dashboards. |


