IN PROGRESS

The idea is to caption images using florence 2 so that a search can be done using the embeddings of the images and the captions of the images.

# Running The Example using docker-compose
To run this example, you need to have docker installed and some knowledge of using docker-compose and basic docker commands will be helpful.<br>
1. Spin up your weaviate instance using the docker-file in this directory by using the command `docker-compose up -d`.
  - The "docker-compose.yml" file has been made using the configurations given on the above mentioned web page.
2. To run the python codes, use the "requirements.txt" file to setup your virtual environment (ex; conda).
  ```sh
  pip install -r app/requirements.txt
  ```
3. After spinning up weaviate, run this command to start the gradio interface.
```sh
python3 app/grui.py --weaviate http://localhost:8080
```
4. Access the ui via `http://localhost:7860/`
  - Before you run, make sure you have access to the images in Sage
>NOTE: Remember to `docker-compose down` when you are done using the example 

# References

- https://weaviate.io/developers/weaviate/manage-data
- https://weaviate.io/developers/weaviate/model-providers/imagebind/embeddings-multimodal
