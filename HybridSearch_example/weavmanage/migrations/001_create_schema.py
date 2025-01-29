from weaviate.classes.config import Configure, Property, DataType, Multi2VecField
import HyperParameters as hp

def run(client):
    """Create the initial schema"""
    # Create a schema to add images, audio, etc.
    # I have used the web pages:
    # https://weaviate.io/developers/weaviate/manage-data
    # https://weaviate.io/developers/weaviate/model-providers/imagebind/embeddings-multimodal
    # to get help on making a suitable schema. You can read the contents of this web page to know more.
    # Define the schema (collection)
    client.collections.create(
        name="HybridSearchExample",
        description="A collection to implement Hybrid Search example",
        properties=[
            Property(name="filename", data_type=DataType.TEXT),
            Property(name="image", data_type=DataType.BLOB),
            Property(name="audio", data_type=DataType.BLOB),
            Property(name="video", data_type=DataType.BLOB),
            Property(name="caption", data_type=DataType.TEXT),  # Caption for keyword search
            Property(name="meta", data_type=DataType.TEXT),
            Property(name="link", data_type=DataType.TEXT),
            # Property(name="timestamp", data_type=DataType.TEXT),
            # Property(name="vsn", data_type=DataType.TEXT),
            # Property(name="node", data_type=DataType.TEXT),
            # Property(name="zone", data_type=DataType.TEXT),
            # Property(name="task", data_type=DataType.TEXT),
            # Property(name="host", data_type=DataType.TEXT),
            # Property(name="job", data_type=DataType.TEXT),
            # Property(name="plugin", data_type=DataType.TEXT),
            # Property(name="host", data_type=DataType.TEXT),
            # Property(name="camera", data_type=DataType.TEXT)
        ],
        vectorizer_config=[
            Configure.NamedVectors.multi2vec_bind(
                name="multi_vector",
                # Define fields for vectorization
                image_fields=[
                    Multi2VecField(name="image", weight=hp.imageWeight)
                ],
                text_fields=[
                    Multi2VecField(name="caption", weight=hp.textWeight)
                ],
                audio_fields=[
                    Multi2VecField(name="audio", weight=hp.audioWeight)
                ],
                video_fields=[
                    Multi2VecField(name="video", weight=hp.videoWeight)
                ],
                vector_index_config=Configure.VectorIndex.hnsw( #https://weaviate.io/developers/weaviate/concepts/vector-index , https://weaviate.io/developers/weaviate/config-refs/schema/vector-index
                    distance_metric=hp.hnsw_dist_metric, #works well to compare images with different attributes such as brightness levels or sizes.
                    dynamic_ef_factor=hp.hnsw_ef_factor,
                    dynamic_ef_max=hp.hsnw_dynamicEfMax,
                    dynamic_ef_min=hp.hsnw_dynamicEfMin,
                    ef=hp.hnsw_ef,
                    ef_construction=hp.hnsw_ef_construction,
                    filter_strategy=hp.hsnw_filterStrategy,
                    flat_search_cutoff=hp.hnsw_flatSearchCutoff,
                    max_connections=hp.hnsw_maxConnections,
                    vector_cache_max_objects=int(hp.hnsw_vector_cache_max_objects),
                    quantizer=hp.hnsw_quantizer,
                )
            )
        ],
        reranker_config=Configure.Reranker.transformers()
    )

    return