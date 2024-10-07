import os
import subprocess


BASE_PITCH_SEQUENCING_PATH_ENV_VAR_NAME= "BASE_PITCH_SEQUENCING_PATH"

def get_project_base_dir() -> str:
    if BASE_PITCH_SEQUENCING_PATH_ENV_VAR_NAME in os.environ:
        return os.environ[BASE_PITCH_SEQUENCING_PATH_ENV_VAR_NAME]
    else:
        # Execute the git command to find the top-level directory
        try:
            return subprocess.check_output(['git', 'rev-parse', '--show-toplevel'], text=True).strip()
        except subprocess.CalledProcessError as e:
            # Handle errors if git command fails (e.g., not a git repository)
            return f"Error: {e}"
        