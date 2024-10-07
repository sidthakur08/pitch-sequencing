import subprocess

def get_gcloud_account_username():
    # Run the gcloud command to get the current account
    result = subprocess.run(["gcloud", "config", "get-value", "account"], capture_output=True, text=True)

    if result.returncode != 0:
        raise Exception("Failed to get gcloud account information: " + result.stderr)

    # Extract the account email
    account_email = result.stdout.strip()

    # Split the email by '@' and take the first part
    username = account_email.split('@')[0]

    return username