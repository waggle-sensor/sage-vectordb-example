'''This file contains the code to run the Benchmark and save the results.'''

import os

#TODO: left of here 02/12/2025, I need to script the queries and generate the results

# Load INQUIRE benchmark dataset from Hugging Face
INQUIRE_DATASET = os.environ.get("INQUIRE_DATASET",  "sagecontinuum/INQUIRE-Benchmark-small")