#!/bin/zsh

# Builds the latest training base image 
# Run this script in this folder for now.
# ./

GIT_SHA_TAG=$(git rev-parse --short HEAD)
TAG=latest

DOCKER_IMAGE="us-central1-docker.pkg.dev/pitch-sequencing/pitch-sequencing-training/base-training-image"

GIT_SHA_IMAGE="${DOCKER_IMAGE}:${GIT_SHA_TAG}"
LATEST_IMAGE="${DOCKER_IMAGE}:${TAG}"


set -ex
docker build -t ${GIT_SHA_IMAGE} .

docker push ${GIT_SHA_IMAGE}

docker tag ${GIT_SHA_IMAGE} ${LATEST_IMAGE}

docker push ${LATEST_IMAGE}

