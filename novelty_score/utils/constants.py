from pathlib import Path

BASE_URL = "https://huggingface.co/datasets/andstor/the_pile_github/resolve/main/data/train/Python/{}"
PARQUET_DIR = Path("parquet_files")
PARQUET_FILE_FORMAT = "part.{}.parquet"

SHM_NAME = "shared_chunk"
CORPUS_DIR = Path("Github_Split")
CORPUS_FILE_FORMAT = "The_Pile_Github_Split_{}.jsonl"
CORPUS_FILES_AMOUNT = 10

ZIP_DIR = Path("zipped")
PLAIN_DIR = Path("raw_files")

# Configurable Parameters
CHUNK_SIZE = 2_000_000
MAX_WORKERS = 8
FUZZ_THRESHOLD = 60
STRIDE_PERCENT = 0.05
