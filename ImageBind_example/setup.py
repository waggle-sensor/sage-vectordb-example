import weaviate

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
        if schema['class']=='BindExample':
            client.schema.delete_class('BindExample')
    # Create a schema to add images, audio, etc.
    # I have used the web page https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/multi2vec-bind
    # to get help on making a suitable schema. You can read the contents of this web page to know more.
    class_obj = {
        "class": "BindExample",
        "description": "A class to implement ImageBind example",
        "vectorizer": "multi2vec-bind",
        "vectorIndexType": "hnsw",
        "moduleConfig": {
            "multi2vec-bind": {
            "textFields": ["text"],
            "imageFields": ["image"],
            "audioFields": ["audio"],
            "videoFields": ["video"],
            },
            "weights": {
            "textFields": [0.4],
            "imageFields": [0.2],
            "audioFields": [0.2],
            "videoFields": [0.2],
            }
        },
        "properties": [
            {
            "dataType": ["string"],
            "name": "text"
            },
            {
            "dataType": ["blob"],
            "name": "image"
            },
            {
            "dataType": ["blob"],
            "name": "audio"
            },
            {
            "dataType": ["blob"],
            "name": "video"
            }
        ]
    }

    client.schema.create_class(class_obj)
    print("Schema class created")

    return client