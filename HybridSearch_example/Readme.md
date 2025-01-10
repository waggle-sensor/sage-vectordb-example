IN PROGRESS

The idea is to caption images using florence 2 so that a search can be done using the embeddings of the images and the captions of the images.

# Running The Example using docker-compose
To run this example, you need to have docker installed and some knowledge of using docker-compose and basic docker commands will be helpful.<br>
1. Spin up your weaviate instance using the docker-compose file in this directory by using the command `docker-compose up -d`.
1. After spinning up weaviate, Access the ui via `http://localhost:7860/`
  - Before you run, make sure you have access to the images in Sage
>NOTE: 
>- Remember to `docker-compose down` when you are done using the example 
> - if your cluster is on another machine, ssh into it using `ssh <client> -L 7860:<EXTERNAL-IP>:7860`. For example, `ssh node-V033 -L 7860:10.31.81.1:7860`

# References

- https://weaviate.io/developers/weaviate/manage-data
- https://weaviate.io/developers/weaviate/model-providers/imagebind/embeddings-multimodal
