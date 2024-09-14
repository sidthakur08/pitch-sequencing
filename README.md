### Dev Setup 
```
conda create -n "pitch-sequencing" python=3.11.9
conda activate pitch-sequencing
pip install -r requirements-dev.txt
pip install -e .
```

### access lfs file

```

Install https://git-lfs.com/

Then do 
cd pitch-sequencing
git lfs install
git lfs pull
```

### Install Docker

https://docs.docker.com/engine/install/

### Setup GCP

Follow https://cloud.google.com/sdk/docs/install

#### Setup Docker to use gcloud auth for GCP 
```
gcloud auth configure-docker us-central1-docker.pkg.dev
```

