import os
import json
import logging
import argparse
from multiprocessing.shared_memory import SharedMemory
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import StringIO
from thefuzz import fuzz
from tqdm import tqdm

CORPUS_DIR = "Github_Split"
CORPUS_FILE_FORMAT = "The_Pile_Github_Split_{}.jsonl"
CORPUS_FILES_AMOUNT = 10
CORPUS_FILES = [
    os.path.join(CORPUS_DIR, CORPUS_FILE_FORMAT.format(part))
    for part in range(CORPUS_FILES_AMOUNT)
]

# Parameters
CHUNK_SIZE = 2_000_000  # Size of each chunk in characters
FUZZ_THRESHOLD = 50  # Minimum fuzzy matching score to be considered a match
STRIDE_PERCENT = 0.05  # Stride percentage for sliding window in fuzzy matching
MAX_WORKERS = 8  # Maximum number of threads for parallel processing
SHM_NAME = "shared_chunk"  # Shared memory name

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def fuzzy_match(test_str, chunk):
    """
    Find the best fuzzy match for a given test string in a chunk of text.
    """
    best_score = 0
    best_match = None
    stride = max(int(len(test_str) * STRIDE_PERCENT), 1)

    for i in range(0, len(chunk) - len(test_str), stride):
        score = fuzz.ratio(chunk[i : i + len(test_str)], test_str)
        if score > best_score:
            best_score = score
            best_match = chunk[i : i + len(test_str)]

    return (best_match, best_score) if best_score >= FUZZ_THRESHOLD else (None, 0)


def fuzzy_match_shared_memory(test_str):
    """
    Find the best fuzzy match for a given test string from shared memory.
    """
    # Attach to the shared memory and decode it
    existing_shm = SharedMemory(name=SHM_NAME)
    chunk = existing_shm.buf[:].tobytes().decode("utf-8")
    existing_shm.close()

    return (test_str, *fuzzy_match(test_str, chunk))


def load_test_data(filename):
    """
    Load test data from a given JSONL file.
    """
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


def load_corpus_data(corpus_files):
    """
    Load corpus data from a list of corpus files.
    """
    corpus_data = []
    for corpus_file in tqdm(corpus_files, desc="Loading corpus files"):
        try:
            with open(corpus_file, "r") as f:
                corpus_data.extend([json.loads(line)["text"] for line in f.readlines()])
        except Exception as e:
            logger.error(f"Error reading {corpus_file}: {e}")
    return corpus_data


def create_corpus_chunks(corpus_data, chunk_size, max_chunks):
    """
    Create fixed-size chunks from the corpus data.
    """
    chunks = []
    i = 0
    while i < len(corpus_data) and (max_chunks is None or len(chunks) < max_chunks):
        str_builder = StringIO()
        while i < len(corpus_data) and str_builder.tell() < chunk_size:
            str_builder.write(corpus_data[i])
            i += 1
        chunks.append(str_builder.getvalue())
    logger.info(f"Created {len(chunks)} corpus chunks")
    return chunks


def search_test_string_in_chunks(test_str, corpus_chunks):
    """
    Search for the best matching string in the corpus chunks for a given test string using parallel processing.
    """
    best_match = None
    best_score = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(fuzzy_match, test_str, chunk): chunk
            for chunk in corpus_chunks
        }

        for future in tqdm(
            as_completed(futures), total=len(futures), desc="Processing chunks"
        ):
            chunk_best_match, chunk_best_score = future.result()
            if chunk_best_score > best_score:
                best_score = chunk_best_score
                best_match = chunk_best_match
                if chunk_best_score == 100:
                    break

    return {"solution": test_str, "closest_solution": best_match, "score": best_score}


def search_multiple_test_strings_in_chunks(test_strings, corpus_chunks):
    """
    Search for the best matching strings in the corpus chunks for a list of test strings using parallel processing.
    """
    overall_results = {
        test_str: {"score": 0, "closest_solution": None} for test_str in test_strings
    }

    for chunk_str in tqdm(corpus_chunks, desc="Processing chunks"):
        # Create shared memory and load the current chunk into it
        chunk_str_bytes = chunk_str.encode("utf-8")
        shm = SharedMemory(name=SHM_NAME, create=True, size=len(chunk_str_bytes))
        shm.buf[: len(chunk_str_bytes)] = chunk_str_bytes

        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {
                executor.submit(fuzzy_match_shared_memory, test_str)
                for test_str in test_strings
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Processing chunks"
            ):
                test_str, best_match, best_score = future.result()
                if best_score > overall_results[test_str]["score"]:
                    overall_results[test_str]["score"] = best_score
                    overall_results[test_str]["closest_solution"] = best_match

        # Clean up shared memory
        shm.close()
        shm.unlink()

    return overall_results


def save_results(results, result_dir, result_file):
    """
    Save the result as a JSONL file.
    """
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, result_file)
    try:
        with open(result_path, "w") as f:
            if isinstance(results, dict):
                json.dump(results, f)
                f.write("\n")
            else:
                for result in results:
                    json.dump(result, f)
                    f.write("\n")
        logger.info(f"Results have been saved successfully to {result_path}")
    except Exception as e:
        logger.error(f"Error saving results to {result_path}: {e}")
        raise


def main(input_path, result_dir, num_corpus_files, num_chunks_to_read):
    logger.info("Reading test file...")
    test_data = load_test_data(input_path)

    logger.info("Loading corpus data...")
    corpus_files = (
        CORPUS_FILES if num_corpus_files is None else CORPUS_FILES[:num_corpus_files]
    )
    corpus_data = load_corpus_data(corpus_files)

    logger.info("Creating corpus chunks...")
    corpus_chunks = create_corpus_chunks(corpus_data, CHUNK_SIZE, num_chunks_to_read)

    logger.info("Searching for solutions in corpus...")
    if isinstance(test_data, list):
        overall_results = search_multiple_test_strings_in_chunks(
            test_data, corpus_chunks
        )
    else:
        overall_results = search_test_string_in_chunks(test_data, corpus_chunks)

    logger.info("Saving results...")
    save_results(overall_results, result_dir, os.path.basename(input_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fuzzy match test strings against corpus data."
    )
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input path containing jsonl test file.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Dir path for putting for jsonl result file in.",
    )
    parser.add_argument(
        "--num_corpus_files",
        type=int,
        default=None,
        help="Number of corpus files to process. Use None for no limit.",
    )
    parser.add_argument(
        "--num_chunks_to_read",
        type=int,
        default=None,
        help="Number of chunks to read for processing. Use None for no limit.",
    )

    args = parser.parse_args()

    if not args.input_path.endswith(".jsonl"):
        raise ValueError("Input file must have a .jsonl extension")
    if args.num_corpus_files and args.num_corpus_files > CORPUS_FILES_AMOUNT:
        raise ValueError(
            f"Number of corpus files must be equal or less than {CORPUS_FILES_AMOUNT}"
        )

    main(
        args.input_path, args.result_dir, args.num_corpus_files, args.num_chunks_to_read
    )
