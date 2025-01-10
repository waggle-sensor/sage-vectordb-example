import weaviate
from weaviate.classes.config import Configure, Property, DataType, Multi2VecField
import HyperParameters as hp

def setup_collection(client):
    '''
    Set up Weaviate collection
    '''

    # V3 VERSION:
    # Checking if caption schema already exists, then delete it
    # current_schemas = client.schema.get()['classes']
    # for schema in current_schemas:
    #     if schema['class']=='HybridSearchExample':
    #         client.schema.delete_class('HybridSearchExample')

    # Check if the collection exists and delete it
    if "HybridSearchExample" in [col.name for col in client.collections.list_all()]:
        client.collections.delete("HybridSearchExample")
        print("Existing collection deleted")

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
            Property(name="timestamp", data_type=DataType.TEXT),
            Property(name="link", data_type=DataType.TEXT),
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
                vector_index_config=Configure.VectorIndex.hnsw(
                # additional fields can be added, like HP such as distance metric. Using defaults
                )
            )
        ],
    )
    print("Collection created")

    return