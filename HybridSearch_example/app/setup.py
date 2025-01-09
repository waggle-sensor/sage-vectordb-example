import weaviate
import HyperParameters as hp

def setup_client():
    '''
    Set up Weaviate client and add class
    '''
    
    client = weaviate.Client("http://localhost:8080")
    #client = weaviate.connect_to_local()
    print("Client created")

    #Checking if caption schema already exists, then delete it
    current_schemas = client.schema.get()['classes']
    for schema in current_schemas:
        if schema['class']=='HybridSearchExample':
            client.schema.delete_class('HybridSearchExample')
    # Create a schema to add images, audio, etc.
    # I have used the web page https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/multi2vec-bind
    # to get help on making a suitable schema. You can read the contents of this web page to know more.
    class_obj = {
        "class": "HybridSearchExample",
        "description": "A class to implement Hybrid Search example",
        "vectorizer": "multi2vec-bind",
        "vectorIndexType": "hnsw",
        "moduleConfig": {
            "multi2vec-bind": {
            "textFields": ["caption"],
            "imageFields": ["image"],
            "audioFields": ["audio"],
            "videoFields": ["video"],
            },
            "weights": {
            "textFields": hp.textWeight,
            "imageFields": hp.imageWeight,
            "audioFields": hp.audioWeight,
            "videoFields": hp.videoWeight,
            }
        },
        "properties": [
            {"name": "filename","dataType": ["string"],},
            {"name": "image","dataType": ["blob"],},
            {"name": "audio","dataType": ["blob"],},
            {"name": "video","dataType": ["blob"],},
            {"name": "caption", "dataType": ["text"]},  # Object caption
            {"name": "timestamp", "dataType": ["text"]},
            {"name": "link", "dataType": ["text"]},
        ]
    }

    client.schema.create_class(class_obj)
    print("Schema class created")

    return client