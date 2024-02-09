# Overview
This is an example created using the weaviate [multi2-vec-clip](https://weaviate.io/developers/weaviate/v1.11.0/retriever-vectorizer-modules/multi2vec-clip.html) module, the weaviate-python client, and the sage data client. The example has a flask based interface where a user can give an image or text as a query and the top 3 images that are similar to the given image or text are returned to the user. (The number of results needed can be altered. You can even fetch only the result with the highest similarity)

# Running The Example
To run this example, you need to have docker installed and some knowledge of using docker-compose and basic docker commands will be helpful.<br>
1. Spin up your weaviate instance using the docker-file in this directory by using the command `docker-compose up -d`.
  - The "docker-compose.yml" file has been made using the configurations given on the above mentioned web page.
2. To run the python codes, set up "config.ini" and use the "requirements.txt" file to setup your virtual environment (ex; conda).
  ```sh
  pip install -r requirements.txt
  ```
3. After spinning up weaviate and getting the environment ready, add data to weaviate by running `python3 data.py`
  - Before you run, make sure you have access to the images in Sage
  - You can also change the query in "data.py"
4. To test if everything is working, run `python3 terminal_test.py`
>TODO: change the terminal_test.py right now it uses default images given by weaviate
5. Now, run `python3 upload.py` to start the flask server and use the frontend.
>NOTE: Remember to `docker-compose down` when you are done using the example 
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
To add different images, change the sage_data_client query in "data.py" and run the "data.py" file.
Similarly, to add more tests, add images to "static/Test" folder.
To add more texts, add them in the list named "texts" in "data.py" file and run that file.
> TODO: change this to reflect how you added images for test <br>

NOTE: I have commented out the part where text can also be added to weaviate. But you can uncomment it and try adding text too. <br>
After adding text, the results may also contain text and images both, for a particular query.<br>
Experimenting with the type of input and observing the different types of results obtained is highly encouraged !!<br>
Have a great time with weaviate !!<br>


# References
- [Weaviate Examples](https://github.com/weaviate/weaviate-examples/tree/main)
