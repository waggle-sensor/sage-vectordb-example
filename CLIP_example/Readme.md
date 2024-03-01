# Overview
This is an example created using the weaviate [multi2-vec-clip](https://weaviate.io/developers/weaviate/v1.11.0/retriever-vectorizer-modules/multi2vec-clip.html) module, the weaviate-python client, and the sage data client. The example has a flask based interface where a user can give an image or text as a query and the top 3 images that are similar to the given image or text are returned to the user. (The number of results needed can be altered. You can even fetch only the result with the highest similarity)

> TODO: edit setup.py to use weaviate python client v4's classes & functions

# Running The Example using docker-compose
To run this example, you need to have docker installed and some knowledge of using docker-compose and basic docker commands will be helpful.<br>
1. Spin up your weaviate instance using the docker-file in this directory by using the command `docker-compose up -d`.
  - The "docker-compose.yml" file has been made using the configurations given on the above mentioned web page.
2. To run the python codes, use the "requirements.txt" file to setup your virtual environment (ex; conda).
  ```sh
  pip install -r requirements.txt
  ```
3. After spinning up weaviate, run `python3 upload.py --weaviate http://localhost:8080` to start the flask server.
4. Access the ui via `http://127.0.0.1:5000/`
  - Before you run, make sure you have access to the images in Sage
>NOTE: Remember to `docker-compose down` when you are done using the example 

# Running The Example on a Kubernetes Cluster
To run this example, you need to have a kubernetes cluster configured and some knowledge of using kubernetes and basic kubectl commands will be helpful. If the example is already deployed jump to step 3.<br>
1. git clone this repo, and travel to the clip example folder
2. Then, run the command `kubectl apply -k kubernetes`
3. Check your services with `kubectl get svc` and retrieve the EXTERNAL-IP for "clip-weaviate-ui"
4. Access the ui via `<EXTERNAL-IP>:5000`
  - Before you access, make sure you have access to the images in Sage 
  >NOTE: if your cluster is on another machine, ssh into it using `ssh <client> -L 5000:<EXTERNAL-IP>:5000`. For example, `ssh node-V033 -L 5000:10.31.81.1:5000`

## Similiarity Measure

**The meaure is returned in the UI**. The measure returned is `certainty` which is calculated by weaviate using `cosine similarity`. View their [FAQ](https://weaviate.io/developers/weaviate/more-resources/faq#q-how-do-i-get-the-cosine-similarity-from-weaviates-certainty) on how they calculate it.

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

# Interesting Results
|Query|Image|Description|
|---|---|---|
|"airplane"|![image](demo_images/airplane.png)| It was able to catch the object in the sky. When I looked up "comet" it also returned this image.|



# References
- [Weaviate Examples](https://github.com/weaviate/weaviate-examples/tree/main)
