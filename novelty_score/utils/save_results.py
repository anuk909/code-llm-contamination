import os
import json
from pathlib import Path

from .logger import logger
from .typing import FuzzyMatchResults, DolosResults


def save_fuzzy_match_results(
    results: FuzzyMatchResults, result_dir: str, result_file: str
) -> None:
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, result_file)
    try:
        with open(result_path, "w") as f:
            for solution, result in results.items():
                f.write(json.dumps({"solution": solution, **result}) + "\n")
        logger.info(f"Results have been saved successfully to {result_path}")
    except Exception as e:
        logger.error(f"Error saving results to {result_path}: {e}")
        raise


def save_dolos_results(
    results: DolosResults, result_dir: Path, result_file: str
) -> None:
    result_dir.mkdir(exist_ok=True)
    results.sort(key=lambda x: x["program_index"])
    result_path = os.path.join(result_dir, result_file)
    with open(result_path, "w") as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")
    logger.info(f"Results have been saved successfully to {result_path}")
