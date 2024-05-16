import os
import json
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm

from utils.parse_arguments import parse_dolos_arguments
from utils.constants import MAX_WORKERS, SOLUTION_TEMP_DIR
from utils.logger import logger
from utils.save_results import save_dolos_results

CANONICAL_SOLUTION_FILE = "canonical_solution.py"
CHUNKS_SOLUTION_DIR_NAME = "chunks_solutions"


def extract_index(path: Path) -> int:
    """Extracts index from the file name."""
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
            stream = os.popen(
                f"dolos run -f terminal --language python {canonical_solution_file} {chunk_solution_file}"
            )
            output = stream.read().split("\n")

            for line in output:
                if "Similarity score:" in line:
                    score = float(line.split(": ")[1])
                    if score > 0:
                        program_results.append({"chunk_index": index, "score": score})
                    break
        except Exception as e:
            logger.error(
                f"Error running dolos on {canonical_solution_file} and {chunk_solution_file}: {e}"
            )
            continue

    sorted_program_results = sorted(
        program_results, key=lambda d: d["score"], reverse=True
    )

    result_dict = {
        "program_index": program_index,
        "sorted_program_results": sorted_program_results,
    }
    return result_dict


def create_solutions_tree(test_file: Path) -> None:
    """Converts test results into necessary file tree for Dolos input."""
    try:
        with test_file.open("r") as f:
            results = [json.loads(line) for line in f]
    except IOError as e:
        logger.error(f"Error reading test file {test_file}: {e}")
        return
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON in {test_file}: {e}")
        return

    SOLUTION_TEMP_DIR.mkdir(exist_ok=True)

    for i, result in enumerate(tqdm(results, desc="Creating Solutions Tree")):
        canonical_solution = result["solution"]

        plain_program_dir = SOLUTION_TEMP_DIR / f"Nobelty_test_{i+1}"
        plain_program_dir.mkdir(exist_ok=True)

        canonical_solution_path = plain_program_dir / CANONICAL_SOLUTION_FILE
        with canonical_solution_path.open("w") as f:
            f.write(canonical_solution)

        chunks_solutions_dir = plain_program_dir / CHUNKS_SOLUTION_DIR_NAME
        chunks_solutions_dir.mkdir(exist_ok=True)

        for chunk_result in result["chunk_results"][:500]:
            chunk_index = chunk_result["chunk_index"]
            chunk_closest_solution = chunk_result["closest_solution"]

            chunk_closest_solution_path = (
                chunks_solutions_dir / f"closest_solution_{chunk_index}.py"
            )
            with chunk_closest_solution_path.open("w") as f:
                f.write(chunk_closest_solution)


def run_dolos_analysis(
    folder_names: List[Path],
) -> List[Any]:
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
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Error processing folder {futures[future]}: {e}")
    return results


def clean_up(dir_path: Path) -> None:
    """Remove all files and directories within the given directory."""
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
    logger.info(f"Cleaned up directory: {dir_path}")


def main(input_path_str: str, result_dir: str) -> None:
    """Main function to orchestrate the Dolos job."""
    input_path: Path = Path(input_path_str)
    if not input_path.exists() or not input_path.is_file():
        logger.error(
            f"The input file {str(input_path)} does not exist or is not a file."
        )
        return

    try:
        create_solutions_tree(input_path)
        program_folders = [f for f in SOLUTION_TEMP_DIR.iterdir() if f.is_dir()]
        results = run_dolos_analysis(program_folders)
        save_dolos_results(
            results, result_dir, "DolosMatch" + os.path.basename(input_path)
        )
    finally:
        pass
        # Always clean up directories at the end
        clean_up(SOLUTION_TEMP_DIR)


if __name__ == "__main__":
    args = parse_dolos_arguments()

    main(args.input_path, args.result_dir)
