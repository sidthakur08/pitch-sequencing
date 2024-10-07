## Summary

Handy Dandy Training Job Launcher Tool. Will package a given training job's directory into a custom docker image, upload to GCP, and create a training job with the given args.

### Usage

Input is the directory of our training job.

```
python scripts/handy/handy.py cmd/transformers
```


### Directory Layout

* main.py (required)
    
    * Entrypoint script for training job.

* job_config.json (required)

    * Input Params to the training script (ie learning rate, etc) and hardware requests (type, GPU, etc). See [job_config.json Layout](#job_configjson-layout) for more details.

* requirements.txt (optional)

    * Any additional python packages not found in usual base training Docker image. 

* Dockerfile (optional)

    * How to build the training Docker image. Should avoid creating in majority of cases.

### job_config.json Layout
Json file

```json
{
    "vertex_training_config": {
        "instance_type": "n1-standard-4",
        "replica_count": "1",
        "gpu_config": {
            "accelerator_type": "NVIDIA_TESLA_T4",
            "accelerator_count": "1"
        },
        "args": [
            "--num_epochs=25",
            "--learning_rate=0.01",
            "--batch_size=64",
            "--input_path=gs://pitch-sequencing/sequence_data/large_sequence_data_cur_opt.csv"
        ]
    }
}
```
