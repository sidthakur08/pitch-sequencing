import gcsfs
import torch 
import torch.nn as nn

def save_model_to_gcs(model: nn.Module, gcs_path: str) -> str:
    fs = gcsfs.GCSFileSystem(project="pitch-sequencing")

    with fs.open(gcs_path, 'wb') as f:
        torch.save(model.state_dict(), f)

    return gcs_path
