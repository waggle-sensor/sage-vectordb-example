## Building for arm64
```sh
docker buildx build --push \
--platform linux/arm64 \
--tag waggle/clip-weaviate-ui:arm64 .
```

## Building for amd64
```sh
docker buildx build --push \
--platform linux/amd64 \
--tag waggle/clip-weaviate-ui .
```