#!/bin/bash
CONTAINER_NAME="wpy"



docker rm -f $CONTAINER_NAME 2>/dev/null

docker run -itd \
    --name $CONTAINER_NAME \
    --gpus all \
    --shm-size=32g \
    --cap-add=SYS_ADMIN \
    -v $(pwd):/workdir \
    -v $(pwd)/env/.zshrc:/root/.zshrc \
    -v $(pwd)/env/.oh-my-zsh:/root/.oh-my-zsh \
    -v $(pwd)/env/.zsh_history:/root/.zsh_history \
    -w /workdir \
    my-pytorch:dev \
    zsh

# 此时直接进入，秒开
docker exec -it $CONTAINER_NAME zsh