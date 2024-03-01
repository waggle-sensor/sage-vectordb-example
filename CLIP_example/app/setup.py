import weaviate

def setup_client(client):
    '''
    Set up Weaviate client and add class
    '''
    #Checking if caption schema already exists, then delete it
    current_schemas = client.schema.get()['classes']
    for schema in current_schemas:
        if schema['class']=='ClipExample':
            client.schema.delete_class('ClipExample')
    # Create a schema to add images
    # I have used the web page https://weaviate.io/developers/weaviate/v1.11.0/retriever-vectorizer-modules/multi2vec-clip.html
    # to get help on making a suitable schema. You can read the contents of this web page to know more.
    class_obj = {
        "class": "ClipExample",
            "description": "A class to implement CLIP example",
            "moduleConfig": {
            "multi2vec-clip": {
            "imageFields": [
                "image"
            ],
            "textFields": [
                "text"
            ],
            "weights": {
                "textFields": [0.7],
                "imageFields": [0.3]
            }
            }
        },
            "vectorIndexType": "hnsw",
            "vectorizer": "multi2vec-clip",
            "properties": [
                {
                "dataType": [
                    "string"
                ],
                "name": "text"
                },
                {
                "dataType": [
                    "blob"
                ],
                "name": "image"
                }
            ]
        }

    client.schema.create_class(class_obj)
    print("Schema class created")

    return client