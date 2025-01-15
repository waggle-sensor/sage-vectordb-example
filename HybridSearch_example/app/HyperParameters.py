'''This file contains the hyper parameters that can be changed to fine tune
the system. 
NOTE: Not all params have been added here. More in depth search must be 
done to find more hyper params that can be altered'''

from weaviate.classes.query import HybridFusion
from weaviate.classes.config import VectorDistances, Configure
from weaviate.collections.classes.config_vector_index import VectorFilterStrategy

#TODO: Grab a big enough sample set to test a real deployment of weaviate with Sage so you can fine tune the HPs
#  NOTE: instead of recreating the db just update the HPs when testing

# 1) Weaviate module multi2vec-bind (Imagebind) weights
textWeight = 0.2
imageWeight = 0.4 # Increase the weighting here so that the embedding is more influenced by the image
audioWeight = 0.2
videoWeight = 0.2

# 2) Hierarchical Navigable Small World (hnsw) for Approximate Nearest Neighbor (ANN) hyperparamaters
# used hsnw since it works well with bigger datasets
# more info: https://weaviate.io/developers/weaviate/config-refs/schema/vector-index#hnsw-indexes
# configuration tips: https://weaviate.io/developers/weaviate/config-refs/schema/vector-index#hnsw-configuration-tips
# helpful article: https://gagan-mehta.medium.com/efficient-resource-understanding-and-planning-in-weaviate-ec673f065e86
hnsw_dist_metric=VectorDistances.COSINE
hnsw_ef=-1 #Balance search speed and recall, Weaviate automatically adjusts the ef value and creates a dynamic ef list when ef is set to -1
hnsw_ef_construction=100 #Balance index search speed and build speed. Changed from 128 to 100
hnsw_maxConnections=50 #Maximum number of connections per element. Changed from 32 to 50
hsnw_dynamicEfMax=500 #Upper bound for dynamic ef
hsnw_dynamicEfMin=200 #Lower bound for dynamic ef. Changed from 100 to 200
hnsw_ef_factor=20 #This setting is only used when hnsw_ef is -1, Sets the potential length of the search list. Changed from 8 to 20
hsnw_filterStrategy=VectorFilterStrategy.ACORN #The filter strategy to use for filtering the search results.
hnsw_flatSearchCutoff=40000 #cutoff to automatically switch to a flat (brute-force) vector search when a filter becomes too restrictive
hnsw_vector_cache_max_objects=1e12 #Maximum number of objects in the memory cache
# Auto Product Quantization (PQ)
#  https://weaviate.io/developers/weaviate/configuration/compression/pq-compression
hnsw_quantizer=Configure.VectorIndex.Quantizer.pq(
    training_limit=500000 #threshold to begin training
)  

# 3) Florence 2 hyperparameters
max_new_tokens=512 #Changed from 1024 to 512
early_stopping=False #Changed from False to True
do_sample=False
num_beams=2 #changed from 3 to 2

# 4) Hybrid Search Query hyperparameters
response_limit=0 #Number of objects to return 
query_alpha=0.5 #An alpha of 1 is a pure vector search, An alpha of 0 is a pure keyword search.
max_vector_distance=0.4 #max accepted distance for the vector search component
concepts_to_avoid=["police", "gun"] # Concepts to avoid
avoid_concepts_force=0 #the strength to avoid the concepts
# fusion algorithm: prepare the scores from each search to be compatible with each other, 
#  so that they can be weighted and added up
fusion_alg=HybridFusion.RELATIVE_SCORE # RELATIVE_SCORE is default from weaviate 1.24
# autocut limits results based on discontinuities
# more info: https://weaviate.io/developers/weaviate/api/graphql/additional-operators#autocut
autocut_jumps=1 #To explicitly disable autocut, set the number of jumps to 0 or a negative value
#NOTE: USE autocut_jumps OR response_limit