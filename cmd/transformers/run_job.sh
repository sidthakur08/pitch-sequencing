#!/bin/zsh

set -e

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" > /dev/null; then
    echo "No active gcloud account. Please authenticate using: gcloud auth login"
    exit 1
fi

# Define variables
USERNAME=$(gcloud config get-value account | cut -d'@' -f1)

REGION_REPO="us-central1-docker.pkg.dev"
PROJECT_ID="pitch-sequencing"
REPO="pitch-sequencing-training"
IMAGE_NAME="transformer"
TAG="${USERNAME}-latest"
REGION="us-central1"
#BUCKET_NAME="your-bucket-name"
JOB_NAME="tranformers_test_run_full_data_size_gpu_1"
# https://cloud.google.com/vertex-ai/docs/training/configure-compute#machine-types for machine types compatabile
INSTANCE_TYPE="n1-standard-4"
# Refer to https://cloud.google.com/vertex-ai/docs/training/configure-compute#specifying_gpus for compatability.
GPU_TYPE="NVIDIA_TESLA_T4"

# Path to your Dockerfile directory (assuming the Dockerfile and training script are in the current directory)
DOCKERFILE_PATH="."

FULL_DOCKER_IMAGE_URI="${REGION_REPO}/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"

set -o
# Build Docker image
echo "Building Docker image \"${FULL_DOCKER_IMAGE_URI}\"..."
docker build -t ${FULL_DOCKER_IMAGE_URI} ${DOCKERFILE_PATH}

# Push the Docker image to Google Artifact Regsitry
echo "Pushing Docker image \"${FULL_DOCKER_IMAGE_URI}\" to GAR..."
docker push ${FULL_DOCKER_IMAGE_URI}

# Submit the training job to Vertex AI
echo "Submitting the job to Vertex AI..."
gcloud beta ai custom-jobs create \
    --region=${REGION} \
    --display-name=${JOB_NAME} \
    --worker-pool-spec="\
machine-type=${INSTANCE_TYPE},\
accelerator-type=${GPU_TYPE},\
accelerator-count=1,\
replica-count=1,\
container-image-uri=${FULL_DOCKER_IMAGE_URI}" 

# For above, the command requires worker spec to be in same string


# Finish
echo "Script completed."