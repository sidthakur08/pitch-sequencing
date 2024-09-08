#!/bin/zsh

set -e

# Define variables
REGION_REPO="us-central1-docker.pkg.dev"
PROJECT_ID="pitch-sequencing"
REPO="pitch-sequencing-training"
IMAGE_NAME="transformer"
TAG="cpu-latest"
REGION="us-central1"
#BUCKET_NAME="your-bucket-name"
JOB_NAME="tranformers_test_run_2"

# Path to your Dockerfile directory (assuming the Dockerfile and training script are in the current directory)
DOCKERFILE_PATH="."

set -o
# Build Docker image
echo "Building Docker image..."
docker build -t ${REGION_REPO}/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG} ${DOCKERFILE_PATH}

# Push the Docker image to Google Container Registry
echo "Pushing Docker image to GCR..."
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
    --worker-pool-spec="machine-type=n4-standard-4,replica-count=1,container-image-uri=${REGION_REPO}/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"

# Finish
echo "Script completed."