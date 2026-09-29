"""One-time setup for ZeroHost vLLM on your own Kaggle account: python setup.py

1. Checks the Kaggle CLI is logged in and reads your username.
2. Asks for your ngrok authtoken and static domain, and generates a vLLM API key.
3. Uploads them as a PRIVATE Kaggle dataset <username>/zerohost-secrets
   (Kaggle drops notebook Secrets on CLI pushes, so the dataset is how they reach the kernel).
4. Writes kernel-metadata.json and optionally pushes the kernel, which starts the server.

Re-run it any time to rotate keys; it uploads a new dataset version.
"""
import getpass
import json
import os
import re
import secrets
import subprocess
import sys
import time

SECRETS_DIR = "zerohost-secrets"


def kaggle(*args, check=True):
    result = subprocess.run([sys.executable, "-m", "kaggle", *args], capture_output=True, text=True)
    if check and result.returncode != 0:
        sys.exit(f"kaggle {' '.join(args)} failed:\n{result.stdout}{result.stderr}")
    return result


def kaggle_username():
    match = re.search(r"username:\s*(\S+)", kaggle("config", "view", check=False).stdout)
    if not match or match.group(1) == "None":
        sys.exit("Kaggle CLI is not logged in. Run: python -m kaggle auth login  (then re-run setup.py)")
    return match.group(1)


def ask(prompt, hidden=False):
    value = (getpass.getpass if hidden else input)(prompt).strip()
    if not value:
        sys.exit("A value is required.")
    return value


def write_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    user = kaggle_username()
    print(f"Kaggle user: {user}\n")

    token = ask("ngrok authtoken (dashboard.ngrok.com > Your Authtoken, input hidden): ", hidden=True)
    domain = ask("ngrok static domain (dashboard.ngrok.com > Domains, e.g. xyz.ngrok-free.app): ")
    domain = domain.removeprefix("https://").removeprefix("http://").rstrip("/")
    api_key = input("vLLM API key (Enter to generate one): ").strip() or secrets.token_urlsafe(32)

    os.makedirs(SECRETS_DIR, exist_ok=True)
    write_json(f"{SECRETS_DIR}/secrets.json",
               {"NGROK_AUTHTOKEN": token, "NGROK_DOMAIN": domain, "VLLM_API_KEY": api_key})
    write_json(f"{SECRETS_DIR}/dataset-metadata.json",
               {"title": "zerohost-secrets", "id": f"{user}/zerohost-secrets", "licenses": [{"name": "unknown"}]})

    exists = kaggle("datasets", "status", f"{user}/zerohost-secrets", check=False).returncode == 0
    if exists:
        kaggle("datasets", "version", "-p", SECRETS_DIR, "-m", "update secrets")
    else:
        kaggle("datasets", "create", "-p", SECRETS_DIR)  # private by default
    print(f"Uploaded private dataset {user}/zerohost-secrets, waiting for Kaggle to process it...")
    for _ in range(30):
        if "ready" in kaggle("datasets", "status", f"{user}/zerohost-secrets", check=False).stdout:
            break
        time.sleep(5)

    with open("kernel-metadata.template.json", encoding="utf-8") as f:
        metadata = json.loads(f.read().replace("KAGGLE_USERNAME", user))
    write_json("kernel-metadata.json", metadata)

    print(f"\nYour endpoint:  https://{domain}/v1")
    print("Model name:     qwen2.5-7b-instruct")
    print(f"API key:        {api_key}")
    print(f"(also saved in {SECRETS_DIR}/secrets.json, which is git-ignored; never commit it)\n")

    if input("Push the kernel and start the server now? [y/N] ").strip().lower() == "y":
        print(kaggle("kernels", "push", "-p", ".").stdout)
        print(f"Follow startup:  python -m kaggle kernels logs -f {metadata['id']}")
        print("The mode comparison takes ~40 min; the log prints PUBLIC_URL when serving starts.")


if __name__ == "__main__":
    main()
