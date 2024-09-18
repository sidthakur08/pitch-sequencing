#!/bin/zsh

set -ex

# Check if gcloud is authenticated
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" > /dev/null; then
    echo "No active gcloud account. Please authenticate using: gcloud auth login"
    exit 1
fi


# Define variables
# Path to your Dockerfile directory (assuming the Dockerfile and training script are in the current directory)
DOCKERFILE_PATH="."
USERNAME=$(gcloud config get-value account | cut -d'@' -f1)
# The main job name tht is based on the directory we're running with.
CMD_PROJECT=$(basename $(realpath ${DOCKERFILE_PATH}))
TIMESTAMP=$(date "+%Y%m%d%H%M%S")

REGION_REPO="us-central1-docker.pkg.dev"
PROJECT_ID="pitch-sequencing"
REPO="pitch-sequencing-training"
IMAGE_NAME="${CMD_PROJECT}"
TAG="${USERNAME}-latest"
REGION="us-central1"
#BUCKET_NAME="your-bucket-name"
JOB_NAME="${CMD_PROJECT}_no_single_pitches_${TIMESTAMP}"
# https://cloud.google.com/vertex-ai/docs/training/configure-compute#machine-types for machine types compatabile
INSTANCE_TYPE="n1-standard-4"
# Refer to https://cloud.google.com/vertex-ai/docs/training/configure-compute#specifying_gpus for compatability.
GPU_TYPE="NVIDIA_TESLA_T4"

FULL_DOCKER_IMAGE_URI="${REGION_REPO}/${PROJECT_ID}/${REPO}/${IMAGE_NAME}:${TAG}"

GLOBAL_TRAINING_RUN_DIRECTORY="gs://pitch-sequencing/training_runs"
################################################################################
# Training Job Args
################################################################################
NUM_EPOCHS=10
LEARNING_RATE=0.001
BATCH_SIZE=64
INPUT_TRAIN_PATH="gs://pitch-sequencing/sequence_data/last_pitch_prediction/large_cur_train.csv"
INPUT_VALIDATION_PATH="gs://pitch-sequencing/sequence_data/last_pitch_prediction/large_cur_validation.csv"
OUTPUT_DIRECTORY="${GLOBAL_TRAINING_RUN_DIRECTORY}/${JOB_NAME}"
LOGGING_DIRECTORY="${OUTPUT_DIRECTORY}/logging"

# Format for gcloud command
TRAINING_ARGS="--num_epochs=${NUM_EPOCHS},--learning_rate=${LEARNING_RATE},--batch_size=${BATCH_SIZE},\
--input_train_path=${INPUT_TRAIN_PATH},--input_validation_path=${INPUT_VALIDATION_PATH},--output_directory=${OUTPUT_DIRECTORY},--logging_directory=${LOGGING_DIRECTORY}"


# Build Docker image
echo "Building Docker image \"${FULL_DOCKER_IMAGE_URI}\"..."
DOCKER_BUILD_CONTEXT=${BASE_PITCH_SEQUENCING_PATH}
docker build -t ${FULL_DOCKER_IMAGE_URI} -f Dockerfile ${DOCKER_BUILD_CONTEXT}
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
container-image-uri=${FULL_DOCKER_IMAGE_URI}"\
    --args="${TRAINING_ARGS}"

# For above, the command requires worker spec to be in same string

set +x

# Finish
echo "\nScript completed. Job Name: ${JOB_NAME}\n"

echo "\nView Tensorboard logs by running \`tensorboard --logdir=${LOGGING_DIRECTORY} && open http://localhost:6006\`\n"