#!/bin/python3

from pathlib import Path
import os


CACHE_DIR = Path("./.cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Set root cache for all HF libs (datasets, hub, transformers, etc.)
_p = str(CACHE_DIR.resolve())
os.environ["HF_HOME"] = _p

print("Set HF home dir to", _p)


from huggingface_hub import HfApi, list_repo_files, create_repo


REPO_ID = "Hibou-Foundation/datian"
LOCAL_DIR = Path("./prepared_dataset")
REPO_TYPE = "dataset"

token = ""

api = HfApi(token=token)


def upload_directory(api, token, local_dir: str, repo_id: str, repo_type: str = "dataset"):
    """
    Uploads all files from `local_dir` to the Hugging Face Hub repository.
    Automatically skips already uploaded files (resumable).
    """

    create_repo(repo_id, repo_type=repo_type, token=token, exist_ok=True)

    local_dir = Path(local_dir)
    if not local_dir.exists():
        raise ValueError(f"Local directory does not exist: {local_dir}")

    print(f"📂 Scanning directory: {local_dir}")
    files_to_upload = [p for p in local_dir.rglob("*") if p.is_file()]
    print(f"Found {len(files_to_upload)} files to check.")

    # 🧠 Get list of already uploaded files
    print(f"🔍 Fetching existing files in repo: {repo_id}")
    existing_files = set(list_repo_files(repo_id=repo_id, repo_type=repo_type))
    print(f"Repo already has {len(existing_files)} files.")

    uploaded_count = 0
    skipped_count = 0
    total_count = len(files_to_upload)

    for fpath in files_to_upload:
        # Normalize path in repo (relative to base directory)
        path_in_repo = str(fpath.relative_to(local_dir)).replace("\\", "/")
        print(f"{skipped_count+uploaded_count}/{total_count} ({skipped_count} skipped)")

        if path_in_repo in existing_files:
            print(f"⏩ Skipping already uploaded: {path_in_repo}")
            skipped_count += 1
            continue

        try:
            print(f"⬆️ Uploading: {path_in_repo} ...")
            api.upload_file(
                path_or_fileobj=str(fpath),
                path_in_repo=path_in_repo,
                repo_id=repo_id,
                repo_type=repo_type,
            )
            uploaded_count += 1
            print(f"✅ Uploaded: {path_in_repo}")
        except Exception as e:
            print(f"❌ Error uploading {fpath}: {e}")
            print("Stopping — rerun this script to resume.")
            break

    print("\n✅ Upload complete.")
    print(f"Uploaded: {uploaded_count}, Skipped: {skipped_count}")


upload_directory(api, token, str(LOCAL_DIR) + "/output", REPO_ID, REPO_TYPE)
