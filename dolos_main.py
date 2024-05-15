import os
import time
import json
import shutil
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm

# Constants
ZIP_DIR = Path("zipped")
PLAIN_DIR = Path("raw_files")
PROCESS_NUM = 8
OUTPUT_FILE = "dolos_results.jsonl"

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def call_dolos(folder_name: str) -> Dict[str, List[Dict[str, float]]]:
    """Runs Dolos on all files in the specified folder and retrieves similarity scores."""
    program_index = folder_name.split("_")[1]
    program_results = []

    folder_path = ZIP_DIR / folder_name

    for zip_file in folder_path.iterdir():
        # Extract index from filename.
        try:
            index = int(zip_file.stem.split("_")[1])
        except (IndexError, ValueError) as e:
            logging.error(f"Failed to parse file index from {zip_file.name}: {e}")
            continue

        try:
            stream = os.popen(f"dolos run -f terminal --language python {zip_file}")
            output = stream.read().split("\n")

            # Extract similarity score
            for line in output:
                if "Similarity score:" in line:
                    score = float(line.split(": ")[1])
                    if score > 0:
                        program_results.append({"chunk_index": index, "score": score})
                    break
        except Exception as e:
            logging.error(f"Error running dolos on {zip_file}: {e}")
            continue

    sorted_program_results = sorted(
        program_results, key=lambda d: d["score"], reverse=True
    )

    result_dict = {
        "program_index": program_index,
        "sorted_program_results": sorted_program_results,
    }

    logging.info(
        f"folder name: {folder_name} finished with {len(sorted_program_results)} results"
    )
    return result_dict


def zip_files(test_file: Path) -> None:
    """Converts test results into necessary zip files for Dolos input."""
    try:
        with test_file.open("r") as f:
            results = [json.loads(line) for line in f]
    except IOError as e:
        logging.error(f"Error reading test file {test_file}: {e}")
        return
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON in {test_file}: {e}")
        return

    PLAIN_DIR.mkdir(exist_ok=True)
    ZIP_DIR.mkdir(exist_ok=True)

    for i, result in enumerate(tqdm(results, desc="Zipping files")):
        start = time.time()
        canonical_solution = result["solution"]
        start_index = i + 1

        plain_program_dir = PLAIN_DIR / f"Copyright_test_{start_index}"
        plain_program_dir.mkdir(exist_ok=True)

        zip_program_dir = ZIP_DIR / f"Problem_{start_index}_zipped"
        zip_program_dir.mkdir(exist_ok=True)

        for chunk_result in result["chunk_results"][:500]:
            chunk_index = chunk_result["chunk_index"]
            chunk_closest_solution = chunk_result["closest_solution"]

            chunk_dir = plain_program_dir / f"chunk_{chunk_index}"
            chunk_dir.mkdir(exist_ok=True)

            chunk_closest_solution_path = chunk_dir / f"closest_solution.py"
            with chunk_closest_solution_path.open("w") as f:
                f.write(chunk_closest_solution)

            canonical_solution_path = chunk_dir / "canonical_solution.py"
            with canonical_solution_path.open("w") as f:
                f.write(canonical_solution)

            chunk_zip_path = os.path.join(zip_program_dir, f"chunk_{chunk_index}")
            shutil.make_archive(str(chunk_zip_path), "zip", str(chunk_dir))

        end = time.time()
        logging.info(f"time taken: {end - start:.2f} seconds")


def run_dolos_analysis(
    folder_names: List[str],
) -> List[Dict[str, List[Dict[str, float]]]]:
    """Run Dolos analysis on multiple folders using ProcessPoolExecutor."""
    results = []
    with ProcessPoolExecutor(max_workers=PROCESS_NUM) as executor:
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
                logging.error(f"Error processing folder {futures[future]}: {e}")
    return results


def save_results(
    results: List[Dict[str, List[Dict[str, float]]]], output_file: str
) -> None:
    """Save results to a JSON Lines file."""
    with open(output_file, "w") as f:
        for result in results:
            json.dump(result, f)
            f.write("\n")
    logging.info(f"Results saved to {output_file}")


def clean_up(dir_path: Path) -> None:
    """Remove all files and directories within the given directory."""
    if dir_path.exists() and dir_path.is_dir():
        shutil.rmtree(dir_path)
    logging.info(f"Cleaned up directory: {dir_path}")


def main(input_path: str) -> None:
    """Main function to orchestrate the Dolos job."""
    input_path = Path(input_path)
    if not input_path.exists() or not input_path.is_file():
        logging.error(
            f"The input file {str(input_path)} does not exist or is not a file."
        )
        return

    try:
        zip_files(input_path)
        folder_names = [f.name for f in ZIP_DIR.iterdir() if f.is_dir()]
        results = run_dolos_analysis(folder_names)
        save_results(results, OUTPUT_FILE)
    finally:
        # Always clean up directories at the end
        clean_up(ZIP_DIR)
        clean_up(PLAIN_DIR)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Dolos Analysis.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input path containing jsonl test file in the correct format.",
    )
    args = parser.parse_args()

    main(args.input_path)
