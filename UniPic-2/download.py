# -*- coding: utf-8 -*-
import os
from pathlib import Path
from huggingface_hub import snapshot_download

# --- Parameters ---
# The repository ID on Hugging Face
REPO_ID = "Skywork/UniPic2-SD3.5M-Kontext-2B"

# The script determines the root directory based on its own location.
SCRIPT_DIR = Path(__file__).resolve().parent

# Define the target folder as a "models" subfolder
# The / operator is the modern way to join paths with pathlib
TARGET_DIR = SCRIPT_DIR / "models"

# --- Script logic ---

def clone_model():
    """
    Clone a Hugging Face repository into the "models" subfolder.
    """
    print(f"Starting repository cloning: {REPO_ID}")
    # Display the final destination path
    print(f"Destination: {TARGET_DIR}")
    print("-" * 30)

    try:
        # Use the new TARGET_DIR for downloading
        snapshot_download(
            repo_id=REPO_ID,
            local_dir=TARGET_DIR,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print("\n" + "-" * 30)
        print(f"✅ Download completed successfully!")
        print(f"The model files are now located in: {TARGET_DIR}")

    except Exception as e:
        print(f"\n❌ An error occurred during the download:")
        print(e)

if __name__ == "__main__":
    # Crucial step: create the "models" folder if it doesn't exist.
    # exist_ok=True prevents an error if the folder already exists.
    print(f"Creating target folder (if needed): {TARGET_DIR}")
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Launch the cloning function
    clone_model()
