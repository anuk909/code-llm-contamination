import os
import requests
from tqdm import tqdm
import pandas as pd
from utils.constants import (
    CORPUS_FILES_AMOUNT,
    CORPUS_DIR,
    CORPUS_FILE_FORMAT,
    BASE_URL,
    PARQUET_DIR,
    PARQUET_FILE_FORMAT,
)
from utils.clean_up import clean_up


def download_parquet_files():
    """
    Download parquet files from the specified URL and save them to the 'parquet_files' directory.
    """
    os.makedirs(PARQUET_DIR, exist_ok=True)

    for part in tqdm(
        range(CORPUS_FILES_AMOUNT), desc="Downloading parquet files", unit="file"
    ):
        file_name = PARQUET_FILE_FORMAT.format(part)
        file_path = PARQUET_DIR / PARQUET_FILE_FORMAT.format(part)

        if file_path.exists():
            continue

        url = BASE_URL.format(file_name)
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for non-2xx status codes

        with open(file_path, "wb") as file:
            file.write(response.content)


def convert_parquet_to_jsonl():
    """
    Convert parquet files to jsonl files and save them to the Corpus directory.
    """
    os.makedirs(CORPUS_DIR, exist_ok=True)

    for part in tqdm(
        range(CORPUS_FILES_AMOUNT),
        desc="Converting parquet files to jsonl",
        unit="file",
    ):
        parquest_file_path = PARQUET_DIR / PARQUET_FILE_FORMAT.format(part)
        jsonl_file_path = CORPUS_DIR / CORPUS_FILE_FORMAT.format(part)

        if jsonl_file_path.exists():
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
    clean_up(PARQUET_DIR)


if __name__ == "__main__":
    main()
