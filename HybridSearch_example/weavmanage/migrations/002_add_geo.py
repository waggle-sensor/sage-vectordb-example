from weaviate.classes.config import Property, DataType

#https://weaviate.io/developers/weaviate/config-refs/datatypes#geocoordinates
def run(client):
    """Add a Geo field"""

    collection = client.collections.get("HybridSearchExample")

    collection.config.add_property(
        Property(
            name="location",
            data_type=DataType.GEO_COORDINATES
        )
    )

    return