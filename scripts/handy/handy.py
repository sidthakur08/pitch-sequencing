#!/usr/bin/env python3

import argparse
import os
import shutil

from pitch_sequencing.constants.project import get_project_base_dir


DEFAULT_BASE_DOCKER_IMAGE="us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-3.py310:latest"
REGION_REPO="us-central1-docker.pkg.dev"
PROJECT_ID="pitch-sequencing"
REPO="pitch-sequencing-training"
REGION="us-central1"


LOCAL_BUILD_PATH=os.path.join(get_project_base_dir(), ".build")

REQUIRED_JOB_FILES=["main.py", "params.json"]
    

def parse_args():
    parser = argparse.ArgumentParser(description="Build a Docker image from a job directory.")

    parser.add_argument('job_target_path', type=str, help='The path to your target job that you want to build and launch.')
    parser.add_argument('--job_suffix', type=str, required=False, help="Optional suffix to give to the generated job name")
    parser.add_argument('--image_tag', type=str, required=False, help="Tag override to give to the built docker image.")
    args = parser.parse_args()

    return args


def push_image(image_name: str):
    pass

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
    """
    dockerfile_path = os.path.join(build_directory, "Dockerfile")
    if os.path.exists(dockerfile_path):
        return False
    
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

ENTRYPOINT ["python", "main.py"]
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


if __name__ == "__main__":
    args = parse_args()
    print(f"hello world {args.job_target_path}")

    # Extract targets from input path. 
    target_directory_path = extract_target_directory(args.job_target_path)
    target_name = extract_target_name(target_directory_path)

    # Prepare temporary docker build directory
    build_directory = prepare_build_directory(target_directory_path)
    print(f"Build directory: {build_directory}")
