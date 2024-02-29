# Overview
This is an example created using the weaviate [multi2vec-bind](https://weaviate.io/developers/weaviate/modules/retriever-vectorizer-modules/multi2vec-bind) module, the weaviate-python client, and the sage data client. The example has a flask based interface where a user can give an image or text as a query and the top 3 images that are similar to the given image or text are returned to the user. (The number of results needed can be altered. You can even fetch only the result with the highest similarity)

# Running The Example
To run this example, you need to have docker installed and some knowledge of using docker-compose and basic docker commands will be helpful.<br>
>NOTE: ImageBind takes more memory and CPUs so you will have to increase Docker's allocation or else the model's container will kill itself
1. Spin up your weaviate instance using the docker-file in this directory by using the command `docker-compose up -d`.
  - The "docker-compose.yml" file has been made using the configurations given on the above mentioned web page.
2. To run the python codes, set up "config.ini" and use the "requirements.txt" file to setup your virtual environment (ex; conda).
  ```sh
  pip install -r requirements.txt
  ```
3. After spinning up weaviate and getting the environment ready, run `python3 upload.py` to start the flask server and use the frontend.
  - Before you run, make sure you have access to the images in Sage
>NOTE: Remember to `docker-compose down` when you are done using the example 

## Viewing Similiarity Measure

**The meaure is returned in your terminal**. The measure returned is `certainty` which is calculated by weaviate using `cosine similarity`. View their [FAQ](https://weaviate.io/developers/weaviate/more-resources/faq#q-how-do-i-get-the-cosine-similarity-from-weaviates-certainty) on how they calculate it.

# What it looks like
Below are screenshots of the results obtained on image and text queries:

1. When the model is given an Image as a query:
![image](demo_images/pride.png)
The similarity in the above images is that all of them contain a pride of lions (group of lions).
<br>

2. When the model is given a Text as a query:
![image](demo_images/college_students.png)
Another example with a text query..
![image](demo_images/businesswoman.png)

# Adding More Data
- To add different images, change the sage_data_client query in the flask frontend and click "load data"
  - you can also clear old data by clicking "clear data" which will delete all images.
- To add more tests, add images to "static/Test" folder and/or add prompts in "terminal_test.py".
NOTE: I have commented out the part where text can also be added to weaviate. But you can uncomment it and try adding text too. <br>
After adding text, the results may also contain text and images both, for a particular query.<br>

# TODO
- vectorize SAGE audio files
- vectorize SAGE Thermal files
- vectorize SAGE IMU data
- add audio search
- add thermal search
- add IMU search
- vectorize SAGE video data
- add video search
- vectorize SAGE depth data
- add depth search
- edit setup.py to use weaviate python client v4's classes & functions