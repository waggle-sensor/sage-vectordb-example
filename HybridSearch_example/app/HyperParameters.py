'''This file contains the hyper parameters that can be changed to fine tune
the system. 
NOTE: Not all params have been added here. More in depth search must be 
done to find more hyper params that can be altered'''

#TODO: Make this into a yaml file

# Weaviate module multi2vec-bind (Imagebind) weights
textWeight = 0.2
imageWeight = 0.4 #add more weight here so that hybrid search can use keyword for text and image for vector
audioWeight = 0.2
videoWeight = 0.2

# Florence 2 hyperparameters
max_new_tokens=512 #Changed from 1024 to 512
early_stopping=False #Changed from False to True
do_sample=False
num_beams=2 #changed from 3 to 2

# Hybrid Search Query hyperparameters
response_limit=5
query_alpha=0.5 #An alpha of 1 is a pure vector search, An alpha of 0 is a pure keyword search.
max_vector_distance=0.4 #Maximum threshold for the vector search component