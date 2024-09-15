#!/bin/zsh

# Set an env variable for the absolute path of our repo.
export BASE_PITCH_SEQUENCING_PATH=$(git rev-parse --show-toplevel) 

echo "export BASE_PITCH_SEQUENCING_PATH=$(git rev-parse --show-toplevel)" >> ~/.zshrc

# Create a conda environment for dev.
conda create -n "pitch-sequencing" python=3.11.9
conda activate pitch-sequencing
pip install -r requirements-dev.txt
pip install -e .

# Configure Docker to authorize against gcloud for specific URI Prefixes.
gcloud auth configure-docker us-central1-docker.pkg.dev
gcloud auth configure-docker
