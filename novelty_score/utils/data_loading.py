import json
from typing import List, Union
from .logger import logger
from pathlib import Path


def load_test_data(filename: Path) -> Union[str, List[str]]:
    try:
        with open(filename, "r") as f:
            lines = f.readlines()
            if len(lines) == 1:
                return json.loads(lines[0])["canonical_solution"]
            else:
                return [json.loads(line)["canonical_solution"] for line in lines]
    except Exception as e:
        logger.error(f"Error loading test data from {filename}: {e}")
        raise


def load_corpus_data(corpus_file: Path) -> List[str]:
    try:
        with open(corpus_file, "r") as f:
            return [json.loads(line)["text"] for line in f.readlines()]
    except Exception as e:
        logger.error(f"Error reading {corpus_file}: {e}")
        raise
