# This is all way too big, fix this.
# CPU ONLY
FROM us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-3.py310:latest

WORKDIR /app

COPY main.py /app/train.py
COPY requirements-dev.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# Why are we going from 14.5 GB to 23 GB?
ENTRYPOINT ["python", "train.py"]