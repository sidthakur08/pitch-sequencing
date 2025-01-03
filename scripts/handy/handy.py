#!/usr/bin/env python3

import argparse
import json
import os
import shutil
import subprocess

from datetime import datetime
from typing import Optional

from pitch_sequencing.constants.gcloud import get_gcloud_account_username
from pitch_sequencing.constants.project import get_project_base_dir

# Docker constants
DEFAULT_BASE_DOCKER_IMAGE="us-central1-docker.pkg.dev/pitch-sequencing/pitch-sequencing-training/base-training-image:latest"
REGION_REPO="us-central1-docker.pkg.dev"
PROJECT_ID="pitch-sequencing"
REPO="pitch-sequencing-training"
REGION="us-central1"
DEFAULT_IMAGE_TAG_TEMPLATE="{username}-latest"
FULL_DOCKER_IMAGE_URI_TEMPLATE="{region_repo}/{project_id}/{repo}/{image_name}:{tag}"

# Training Params
GLOBAL_TRAINING_RUN_DIRECTORY="gs://pitch-sequencing/training_runs"
TRAINING_OUTPUT_TEMPLATE=GLOBAL_TRAINING_RUN_DIRECTORY+"/{job_name}"


# Build constants
LOCAL_BUILD_PATH=os.path.join(get_project_base_dir(), ".build")

MAIN_PY_FILE_NAME="main.py"
JOB_CONFIG_JSON_FILE_NAME="job_config.json"

REQUIRED_JOB_FILES=[MAIN_PY_FILE_NAME, JOB_CONFIG_JSON_FILE_NAME]
    

def parse_args():
    parser = argparse.ArgumentParser(description="Build a Docker image from a job directory.")

    parser.add_argument('job_target_path', type=str, help='The path to your target job that you want to build and launch.')
    parser.add_argument('--job_suffix', type=str, required=False, help="Optional suffix to give to the generated job name")
    parser.add_argument('--image_tag', type=str, required=False, help="Tag override to give to the built docker image.")
    args = parser.parse_args()

    return args


def extract_target_directory(input_path: str) -> str:
    """
    Extracts the target directory path from the input path. 
    
    If a directory is given, returns that directory.
    If a file path is given, returns the parent directory that contains that file.
    """

    # Clean any excess 
    input_path = input_path.strip("/")
    if os.path.isdir(input_path):
        return input_path
    else:
        return os.path.dirname(input_path)
    
def extract_target_name(target_directory: str) -> str:
    return os.path.basename(target_directory)

def validate_required_job_files_exist(build_directory: str):
    """
    Ensures all expected files are present in build directory. If not, raise exception.
    """
    for required_file in REQUIRED_JOB_FILES:
        if not os.path.exists(os.path.join(build_directory, required_file)):
            raise Exception(f"Required file {required_file} missing from build directory")
        
def maybe_generate_empty_requirements_file(build_directory: str) -> bool:
    """
        Checks if a requirements.txt file is present in build directory, if not create an empty one.
        
        Returns true if one was created.
    """
    requirements_path = os.path.join(build_directory, "requirements.txt")
    if os.path.exists(requirements_path):
        return False
    
    with open(requirements_path, 'w'):
        pass

    return True

def maybe_generate_dockerfile(build_directory: str) -> bool:
    """
    Checks if a Dockerfile is present in build directory, if not create a generic templated one.

    Returns true if one was created.
    """
    dockerfile_path = os.path.join(build_directory, "Dockerfile")
    if os.path.exists(dockerfile_path):
        return False

    # We will set the docker build context to the project's root directory
    # So we need our build directory's relative path to that when passing in. 
    rel_target_build_directory = os.path.relpath(build_directory, get_project_base_dir())
    
    default_dockerfile = f"""
FROM {DEFAULT_BASE_DOCKER_IMAGE}
RUN apt-get update

WORKDIR /app

COPY {rel_target_build_directory}/requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY {rel_target_build_directory} /app

COPY . /app
RUN pip install -e .

ENTRYPOINT ["python", "{MAIN_PY_FILE_NAME}"]
"""
    
    with open(dockerfile_path, 'w') as f:
            f.write(default_dockerfile)

    return True


def prepare_build_directory(target_directory_path: str) -> str:
    abs_target_path = os.path.abspath(target_directory_path)
    # Determine the full relative path of our target inside of our project.
    rel_path = os.path.relpath(abs_target_path, get_project_base_dir())

    # Create a directory for our build target in our project's build directory.
    target_build_directory = os.path.join(LOCAL_BUILD_PATH, rel_path)
    rel_target_build_directory = os.path.relpath(LOCAL_BUILD_PATH, get_project_base_dir())

    # Clean up any previous builds for this target.
    if os.path.exists(target_build_directory):
        shutil.rmtree(target_build_directory)

    # Create new clean build directory.
    os.makedirs(target_build_directory)

    # Copy all files from the source directory to the build directory
    for item in os.listdir(abs_target_path):
        source_item = os.path.join(abs_target_path, item)
        destination_item = os.path.join(target_build_directory, item)
        if os.path.isfile(source_item):
            shutil.copy(source_item, destination_item)
        elif os.path.isdir(source_item):
            shutil.copytree(source_item, destination_item)

    # Ensure all of our expected files are present.
    validate_required_job_files_exist(target_build_directory)
    maybe_generate_empty_requirements_file(target_build_directory)
    maybe_generate_dockerfile(target_build_directory)

    return target_build_directory

def build_docker_image(target_name: str, build_directory: str, image_tag_override: Optional[str]) -> str:
    """
    Builds a Docker image where the Name of the image is built using target_name.
    Leverages the build_directory's Dockerfile but sets the Docker Build context to the root project directory.
    If image_tag_override is given, use that. Otherwise just use the {GCP username}-latest for tag.

    Returns the full docker image URI.
    """
    image_tag = ""
    if image_tag_override is not None:
        image_tag = image_tag_override
    else:
        image_tag = DEFAULT_IMAGE_TAG_TEMPLATE.format(username=get_gcloud_account_username())
    
    docker_image_uri = FULL_DOCKER_IMAGE_URI_TEMPLATE.format(region_repo=REGION_REPO, project_id=PROJECT_ID, repo=REPO, image_name=target_name, tag=image_tag)
    print(f"Building docker image {docker_image_uri}")
    
    try:
        # Constructing the docker build command
        command = [
            "docker", "build", "-t", docker_image_uri, "-f", 
            os.path.join(build_directory, "Dockerfile"), get_project_base_dir()
        ]
        
        # Execute the docker build command
        result = subprocess.run(command, check=True, text=True)
        
    except subprocess.CalledProcessError as e:
        # Handle errors in the subprocess
        print("Failed to build Docker image:")
        raise e

    return docker_image_uri

def push_docker_image(docker_image_uri: str):
    try:
        # Constructing the docker build command
        command = [
            "docker", "push", docker_image_uri
        ]

        # Execute the docker build command
        result = subprocess.run(command, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Failed to push Docker Image")
        raise e

def generate_job_name(build_directory: str, job_suffix: Optional[str]) -> str:
    """
    Generates the name for the training job. Extracts the basename from the build directory and appends the timestamp.
    If job_suffix is provided, append that as well.

    Returns {basename}_training_job_{timestamp}_{job_suffix}
    """
    basename = os.path.basename(build_directory)
    
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')

    job_name = f"{basename}_training_job_{timestamp}"

    if job_suffix is not None:
        job_name = job_name + f"_{job_suffix}"

    return job_name

def build_worker_pool_spec(docker_image_uri: str, vertex_config) -> str:
    """
    Builds the command line key-value pairs for GCP Vertex AI's worker-pool-spec
    https://cloud.google.com/sdk/gcloud/reference/ai/custom-jobs/create#--worker-pool-spec

    Uses a json dictionary config defined in README.
    """
    worker_pool_spec = []
    worker_pool_spec.append(f"machine-type={vertex_config['instance_type']}")
    worker_pool_spec.append(f"replica-count={vertex_config['replica_count']}")
    if "gpu_config" in vertex_config:
        gpu_config = vertex_config["gpu_config"]
        worker_pool_spec.append(f"accelerator-type={gpu_config['accelerator_type']}")
        worker_pool_spec.append(f"accelerator-count={gpu_config['accelerator_count']}")
    else:
        print("No GPU config given!")
    
    worker_pool_spec.append(f"container-image-uri={docker_image_uri}")

    return ",".join(worker_pool_spec)

def build_job_args(vertex_config, output_directory: str) -> str:
    """
    Takes args from input config and builds other generated args for python training script.
    https://cloud.google.com/sdk/gcloud/reference/ai/custom-jobs/create#--args 
    """
    args = vertex_config['args']

    logging_directory = f"{output_directory}/logging"

    args.append(f"--output_directory={output_directory}")
    args.append(f"--logging_directory={logging_directory}")

    return ",".join(args)


def build_vertex_create_command(job_name: str, docker_image_uri: str, vertex_config, output_directory: str):
    """
    Builds the create job comand described here:
    https://cloud.google.com/sdk/gcloud/reference/ai/custom-jobs/create
    """
    worker_pool_spec = build_worker_pool_spec(docker_image_uri, vertex_config)
    job_args = build_job_args(vertex_config, output_directory)

    create_command = ["gcloud", "beta", "ai", "custom-jobs", "create"]
    create_command.append(f"--region={REGION}")
    create_command.append(f"--display-name={job_name}")
    create_command.append(f"--worker-pool-spec={worker_pool_spec}")
    create_command.append(f"--args={job_args}")

    return create_command



def launch_vertex_job_from_config(build_directory: str, docker_image_uri: str, job_suffix: Optional[str]):
    """
    Gathers input configs and build state and launches a GCP Vertex AI training job using CLI:
    https://cloud.google.com/sdk/gcloud/reference/ai/custom-jobs/create

    TODO(kaelen): Figure out how to use their Python API instead.
    """
    with open(os.path.join(build_directory, JOB_CONFIG_JSON_FILE_NAME), 'r') as file:
        job_config_data = json.load(file)

    job_name = generate_job_name(build_directory, job_suffix)
    output_directory = TRAINING_OUTPUT_TEMPLATE.format(job_name=job_name)

    vertex_config = job_config_data["vertex_training_config"]

    create_command = build_vertex_create_command(job_name, docker_image_uri, vertex_config, output_directory)

    try:
        # Execute the docker build command
        result = subprocess.run(create_command, check=True, text=True)
    except subprocess.CalledProcessError as e:
        print("Failed to push Docker Image")
        raise e


if __name__ == "__main__":
    args = parse_args()

    # Extract targets from input path. 
    target_directory_path = extract_target_directory(args.job_target_path)
    target_name = extract_target_name(target_directory_path)

    # Prepare temporary docker build directory
    build_directory = prepare_build_directory(target_directory_path)
    print(f"Build directory: {build_directory}")

    docker_image_name = build_docker_image(target_name, build_directory, args.image_tag)
    print(f"Successfully built {docker_image_name}")

    push_docker_image(docker_image_name)
    print("Succesfully pushed Docker Image to GCP")

    launch_vertex_job_from_config(build_directory, docker_image_name, args.job_suffix)
