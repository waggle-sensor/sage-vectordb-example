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

# 2) Agent Hyperparameters
model = "llama3.2" # make sure ollama pulled the model already
recursion_limit=25 #limit of recursions the agent can do in the workflow
# Define a system prompt that tells the agent its role
# SYSTEM_PROMPT = """ 
# You are SAGE Image Search Agent, an intelligent assistant that can search through images from an application called SAGE.
# Always return the Image links.
# If the user does not require image search, answer normally.
# When a user requests an image search, you must respond with a command in the following format:
# <search query>
# For example, if a user asks "Show me images of Hawaii", you should respond with:
# Hawaii
# After the image search tool returns results, incorporate them into your final answer.
# """
SYSTEM_PROMPT = """ 
You are SAGE Search Agent, an intelligent assistant that can search through Images and device data.
Always return the Image links.
If the user does not require image or node search, answer normally.
When a user requests an image search, you must respond with a command in the following format:
<search query>
For example, if a user asks "Show me images of Hawaii", you should respond with:
Hawaii
When a user requests a node search, you must enter the vsn in the node_search_tool.
For example, if a user asks "Show me details on W073", you should use the node_search_tool with a query of W073.
The node search responds with the hardware of the node, what devices are connected to the node, capabilities of the node, and other metadata.
After the image or node search tool returns results, incorporate them into your final answer.
Sage is a distributed software-defined sensor network and a Geographically distributed sensor systems that include cameras, microphones, and 
weather and air quality stations. 
The most common users have included:
Domain scientists interested in developing edge AI applications.
Users interested in sensor and application-produced datasets.
Cyberinfrastructure researchers interested in platform research.
Domain scientists interested in adding new sensors and deploying nodes to answer specific science questions.
"""