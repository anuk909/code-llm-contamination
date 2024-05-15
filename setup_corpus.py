import os
import requests
from tqdm import tqdm
import pandas as pd

BASE_URL = "https://huggingface.co/datasets/andstor/the_pile_github/resolve/main/data/train/Python/{}"
PARQUET_DIR = "parquet_files"
PARQUET_FILE_FORMAT = "part.{}.parquet"
JSONL_DIR = "Github_Split"
JSONL_FILE_FORMAT = "The_Pile_Github_Split_{}.jsonl"
NUM_PARTS = 10


def download_parquet_files():
    """
    Download parquet files from the specified URL and save them to the 'parquet_files' directory.
    """
    os.makedirs(PARQUET_DIR, exist_ok=True)

    for part in tqdm(range(NUM_PARTS), desc="Downloading parquet files", unit="file"):
        file_name = PARQUET_FILE_FORMAT.format(part)
        file_path = os.path.join(PARQUET_DIR, file_name)

        if os.path.exists(file_path):
            continue

        url = BASE_URL.format(file_name)
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for non-2xx status codes

        with open(file_path, "wb") as file:
            file.write(response.content)


def convert_parquet_to_jsonl():
    """
    Convert parquet files to jsonl files and save them to the 'GithubSplit' directory.
    """
    os.makedirs(JSONL_DIR, exist_ok=True)

    for part in tqdm(
        range(NUM_PARTS), desc="Converting parquet files to jsonl", unit="file"
    ):
        parquest_file_path = os.path.join(PARQUET_DIR, PARQUET_FILE_FORMAT.format(part))
        jsonl_file_path = os.path.join(JSONL_DIR, JSONL_FILE_FORMAT.format(part))

        if os.path.exists(jsonl_file_path):
            continue

        df = pd.read_parquet(parquest_file_path)
        df.to_json(
            jsonl_file_path,
            orient="records",
            lines=True,
        )


def main():
    download_parquet_files()
    convert_parquet_to_jsonl()


if __name__ == "__main__":
    main()
