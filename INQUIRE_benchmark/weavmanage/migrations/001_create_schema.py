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
        name="INQUIRE",
        description="A collection to test our set up using INQUIRE",
        properties=[
            Property(name="inat24_image_id", data_type=DataType.NUMBER),
            Property(name="inat24_file_name", data_type=DataType.TEXT),
            Property(name="query", data_type=DataType.TEXT),
            Property(name="query_id", data_type=DataType.NUMBER),
            Property(name="image", data_type=DataType.BLOB),
            Property(name="audio", data_type=DataType.BLOB),
            Property(name="video", data_type=DataType.BLOB),
            Property(name="caption", data_type=DataType.TEXT),  # Caption for keyword search
            Property(name="link", data_type=DataType.TEXT),
            Property(name="relevant", data_type=DataType.TEXT),
            Property(name="clip_score", data_type=DataType.TEXT),
            Property(name="supercategory", data_type=DataType.TEXT),
            Property(name="category", data_type=DataType.TEXT),
            Property(name="iconic_group", data_type=DataType.TEXT),
            Property(name="inat24_species_id", data_type=DataType.NUMBER),
            Property(name="inat24_species_name", data_type=DataType.TEXT),
            Property(name="location_uncertainty", data_type=DataType.NUMBER),
            Property(name="date", data_type=DataType.DATE),
            Property(name="location", data_type=DataType.GEO_COORDINATES)
        ],
        vectorizer_config=[
            Configure.NamedVectors.multi2vec_bind(
                name="search",
                vectorize_collection_name= False,
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