import os
import requests
from tqdm import tqdm
import pyarrow.parquet as pq
import json
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
        parquet_file_path = PARQUET_DIR / PARQUET_FILE_FORMAT.format(part)
        jsonl_file_path = CORPUS_DIR / CORPUS_FILE_FORMAT.format(part)

        if parquet_file_path.exists() or jsonl_file_path.exists():
            continue

        url = BASE_URL.format(file_name)
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for non-2xx status codes

        with open(parquet_file_path, "wb") as file:
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
        parquet_file_path = PARQUET_DIR / PARQUET_FILE_FORMAT.format(part)
        jsonl_file_path = CORPUS_DIR / CORPUS_FILE_FORMAT.format(part)

        if jsonl_file_path.exists():
            continue

        table = pq.read_table(parquet_file_path)
        records = table.to_pylist()

        with open(jsonl_file_path, "w") as f:
            for record in records:
                json_record = json.dumps(record)
                f.write(json_record + "\n")


def main():
    download_parquet_files()
    convert_parquet_to_jsonl()
    clean_up(PARQUET_DIR)


if __name__ == "__main__":
    main()
