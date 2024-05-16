import os
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from utils.parse_arguments import parse_dolos_arguments
from utils.constants import (
    MAX_WORKERS,
    SOLUTION_TEMP_DIR,
    CHUNKS_SOLUTION_DIR_NAME,
    CANONICAL_SOLUTION_FILE,
)
from utils.logger import logger
from utils.save_results import save_dolos_results


def extract_index(path: Path) -> int:
    """Extracts the index from the file name."""
    try:
        return int(path.stem.rsplit("_", 1)[1])
    except (IndexError, ValueError) as e:
        logger.error(f"Failed to parse index from {path.name}: {e}")
        return -1


def call_dolos(program_dir: Path) -> Dict[str, Any]:
    """Runs Dolos on all files in the specified folder and retrieves similarity scores."""
    program_index = extract_index(program_dir)
    program_results = []

    canonical_solution_file = program_dir / CANONICAL_SOLUTION_FILE
    chunks_solutions_dir = program_dir / CHUNKS_SOLUTION_DIR_NAME

    for chunk_solution_file in chunks_solutions_dir.iterdir():
        index = extract_index(chunk_solution_file)
        try:
            command = f"dolos run -f terminal --language python {canonical_solution_file} {chunk_solution_file}"
            with os.popen(command) as stream:
                output = stream.read().split("\n")

            for line in output:
                if "Similarity score:" in line:
                    score = float(line.split(": ")[1])
                    if score > 0:
                        program_results.append({"chunk_index": index, "score": score})
                    break
        except Exception as e:
            logger.error(
                f"Error running Dolos on {canonical_solution_file} and {chunk_solution_file}: {e}"
            )
            continue

    program_results.sort(key=lambda d: d["score"], reverse=True)

    return {
        "program_index": program_index,
        "program_best_matches": program_results,
    }


def create_solutions_tree(test_file: Path) -> None:
    """Converts test results into the necessary file tree for Dolos input."""
    try:
        with test_file.open("r") as f:
            results = [json.loads(line) for line in f]
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Error reading or parsing test file {test_file}: {e}")
        raise

    SOLUTION_TEMP_DIR.mkdir(exist_ok=True)

    for i, result in enumerate(tqdm(results, desc="Creating Solutions Tree")):
        plain_program_dir = SOLUTION_TEMP_DIR / f"Nobelty_test_{i + 1}"
        plain_program_dir.mkdir(exist_ok=True)

        canonical_solution_path = plain_program_dir / CANONICAL_SOLUTION_FILE
        with canonical_solution_path.open("w") as f:
            f.write(result["solution"])

        chunks_solutions_dir = plain_program_dir / CHUNKS_SOLUTION_DIR_NAME
        chunks_solutions_dir.mkdir(exist_ok=True)

        for chunk_result in result["chunk_results"][:500]:
            chunk_path = (
                chunks_solutions_dir
                / f"closest_solution_{chunk_result['chunk_index']}.py"
            )
            with chunk_path.open("w") as f:
                f.write(chunk_result["closest_solution"])


def run_dolos_analysis(folder_names: List[Path]) -> List[Any]:
    """Run Dolos analysis on multiple folders using ProcessPoolExecutor."""
    results = []

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(call_dolos, folder): folder for folder in folder_names
        }

        for future in tqdm(
            as_completed(futures),
            total=len(folder_names),
            desc="Running Dolos Analysis",
        ):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Error processing folder {futures[future]}: {e}")
                raise

    return results


def clean_up(dir_path: Path) -> None:
    """Remove all files and directories within the given directory."""
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
    logger.info(f"Cleaned up directory: {dir_path}")


def main(input_path_str: str, result_dir: str) -> None:
    """Main function to orchestrate the Dolos job."""
    input_path = Path(input_path_str)

    if not input_path.exists() or not input_path.is_file():
        logger.error(f"The input file {input_path} does not exist or is not a file.")
        raise

    try:
        create_solutions_tree(input_path)

        program_folders = [f for f in SOLUTION_TEMP_DIR.iterdir() if f.is_dir()]
        results = run_dolos_analysis(program_folders)

        save_dolos_results(results, result_dir, f"DolosMatch_{input_path.stem}")
    finally:
        clean_up(SOLUTION_TEMP_DIR)


if __name__ == "__main__":
    args = parse_dolos_arguments()
    main(args.input_path, args.result_dir)
