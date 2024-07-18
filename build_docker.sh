#!/bin/bash

USER_NAME=robot
USER_ID=$(id -u)
USER_GID=$(id -g)
BASE_NAME=$(basename "$PWD")

docker build -t thesis/$BASE_NAME \
             --build-arg USER_NAME=$USER_NAME \
             --build-arg USER_ID=$USER_ID \
             --build-arg USER_GID=$USER_GID .
