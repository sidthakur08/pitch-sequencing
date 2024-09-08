# This is all way too big, fix this.
# CPU ONLY
#FROM us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-3.py310:latest
FROM python:3.11.9-slim-bullseye

RUN apt-get update

WORKDIR /app

COPY main.py /app/train.py
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Why are we going from 14.5 GB to 23 GB?
ENTRYPOINT ["python", "train.py"]