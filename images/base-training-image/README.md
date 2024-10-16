# Summary

Our base training image that should be used for all training jobs.

Uses `us-docker.pkg.dev/vertex-ai/training/pytorch-xla.2-3.py310:latest` for GPU pytorch and additionally
installs tensorflow and tensorboard to enable gcs file savings in various packages. (super annoying I know.)
