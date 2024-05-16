import os
import json
from .logger import logger


def save_results(results, result_dir, result_file, is_single):
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, result_file)
    try:
        with open(result_path, "w") as f:
            if is_single:
                f.write(json.dumps(results))
            else:
                for solution, result in results.items():
                    f.write(json.dumps({"solution": solution, **result}) + "\n")
        logger.info(f"Results have been saved successfully to {result_path}")
    except Exception as e:
        logger.error(f"Error saving results to {result_path}: {e}")
        raise
