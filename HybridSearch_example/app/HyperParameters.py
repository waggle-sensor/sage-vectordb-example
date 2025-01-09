#TODO: Make this into a yaml file

# Weaviate module multi2vec-bind (Imagebind) weights
textWeight = [0.4]
imageWeight = [0.2]
audioWeight = [0.2]
videoWeight = [0.2]

# Florence 2 hyperparameters
max_new_tokens=512 #Changed from 1024 to 512
early_stopping=False #Changed from False to True
do_sample=False
num_beams=2 #changed from 3 to 2