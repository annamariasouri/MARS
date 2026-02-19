"""
download_model.py
=================
Downloads the MARS Random Forest chlorophyll model (~3 GB) from Google Drive
and saves it to  data/models/rf_chl_retrained.pkl.

Usage
-----
    pip install gdown
    python download_model.py
"""

import os
import sys

GDRIVE_FILE_ID = "1ZnOU1hdEDRIz3DlQQlGsGwBtcAeTkgnO"
MODEL_DIR  = os.path.join("data", "models")
MODEL_PATH = os.path.join(MODEL_DIR, "rf_chl_retrained.pkl")


def main():
    try:
        import gdown
    except ImportError:
        print("gdown is not installed. Run:  pip install gdown")
        sys.exit(1)

    os.makedirs(MODEL_DIR, exist_ok=True)

    if os.path.exists(MODEL_PATH):
        size_gb = os.path.getsize(MODEL_PATH) / 1e9
        print(f"Model already exists at {MODEL_PATH}  ({size_gb:.2f} GB) — skipping download.")
        return

    url = f"https://drive.google.com/uc?id={GDRIVE_FILE_ID}"
    print(f"Downloading model from Google Drive → {MODEL_PATH}")
    print("This file is ~3 GB; download time depends on your connection.")

    gdown.download(url, MODEL_PATH, quiet=False, fuzzy=True)

    if os.path.exists(MODEL_PATH):
        size_gb = os.path.getsize(MODEL_PATH) / 1e9
        print(f"\nDone. Model saved at {MODEL_PATH}  ({size_gb:.2f} GB)")
    else:
        print("\nDownload failed — file not found after gdown completed.")
        print("If the file is large, try:  gdown --fuzzy '<share_url>'")
        sys.exit(1)


if __name__ == "__main__":
    main()
