'''This file contains the hyper parameters that can be changed to fine tune
the system. 
NOTE: Not all params have been added here. More in depth search must be 
done to find more hyper params that can be altered'''
#NOTE: The hyperparameters will be split up based on what microservice it corresponds to. Or I can
#   make all the microservices apart of the same deployment so the HPs continue to be easily managed
#   and don't get split up.

from weaviate.classes.query import HybridFusion

#TODO: Grab a big enough sample set to test a real deployment of weaviate with Sage so you can fine tune the HPs
#  NOTE: instead of recreating the db just update the HPs when testing

# 1) Hybrid Search Query hyperparameters
response_limit=0 #Number of objects to return 
query_alpha=0.4 #An alpha of 1 is a pure vector search, An alpha of 0 is a pure keyword search.
max_vector_distance=0.4 #max accepted distance for the vector search component
near_text_certainty=0.7 #The minimum similarity score to return. If not specified, the default certainty specified by the server is used.
#NOTE: USE max_vector_distance OR near_text_certainty
concepts_to_avoid=["police", "gun"] # Concepts to avoid
avoid_concepts_force=0 #the strength to avoid the concepts
# fusion algorithm: prepare the scores from each search to be compatible with each other, 
#  so that they can be weighted and added up
fusion_alg=HybridFusion.RELATIVE_SCORE # RELATIVE_SCORE is default from weaviate 1.24
# autocut limits results based on discontinuities
# more info: https://weaviate.io/developers/weaviate/api/graphql/additional-operators#autocut
autocut_jumps=1 #To explicitly disable autocut, set the number of jumps to 0 or a negative value
#NOTE: USE autocut_jumps OR response_limit

# 1) Agent Hyperparameters
recursion_limit=2