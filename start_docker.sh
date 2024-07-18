#!/bin/bash

USER_NAME=robot
BASE_NAME=$(basename "$PWD")
DIR_NAME=$(dirname "$PWD")

docker run --name $BASE_NAME \
           --gpus all \
           -it \
           -v /tmp/.X11-unix:/tmp/.X11-unix \
           -v $DIR_NAME/$BASE_NAME/docker_volume:/home/$USER_NAME/docker_volume \
           -v $DIR_NAME/shared_docker_volume:/home/$USER_NAME/shared_docker_volume \
           -e DISPLAY=$DISPLAY \
           --rm \
           thesis/$BASE_NAME
