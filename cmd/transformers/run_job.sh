#!/bin/zsh

set -e

# Define variables
REGION_REPO="us-central1-docker.pkg.dev"
PROJECT_ID="pitch-sequencing"
REPO="pitch-sequencing-training"
IMAGE_NAME="transformer"
TAG="latest"
REGION="us-central1"
#BUCKET_NAME="your-bucket-name"
JOB_NAME="tranformers_test_run_full_data_size_gpu_1"
# https://cloud.google.com/vertex-ai/docs/training/configure-compute#machine-types for machine types compatabile
INSTANCE_TYPE="n1-standard-4"
# Refer to https://cloud.google.com/vertex-ai/docs/training/configure-compute#specifying_gpus for compatability.
GPU_TYPE="NVIDIA_TESLA_T4"

# Path to your Dockerfile directory (assuming the Dockerfile and training script are in the current directory)
DOCKERFILE_PATH="."

set -o
# Build Docker image
echo "Building Docker image..."
docker build -t ${REGION_REPO}/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG} ${DOCKERFILE_PATH}

# Push the Docker image to Google Artifact Regsitry
echo "Pushing Docker image to GAR..."
docker push ${REGION_REPO}/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" > /dev/null; then
    echo "No active gcloud account. Please authenticate using: gcloud auth login"
    exit 1
fi

# Submit the training job to Vertex AI
echo "Submitting the job to Vertex AI..."
gcloud beta ai custom-jobs create \
    --region=${REGION} \
    --display-name=${JOB_NAME} \
    --worker-pool-spec="machine-type=${INSTANCE_TYPE},\accelerator-type=${GPU_TYPE},accelerator-count=1,replica-count=1,container-image-uri=${REGION_REPO}/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"


# Finish
echo "Script completed."