import os
import json
import logging
import argparse
from multiprocessing import shared_memory, Pool
from io import StringIO
from thefuzz import fuzz
from tqdm import tqdm

CORPUS_DIR = "Github_Split"
CORPUS_FILES = [
    os.path.join(CORPUS_DIR, f"The_Pile_Github_Split_{i}.jsonl") for i in range(10)
]

CHUNK_SIZE = 2_000_000  # Chunk size by character
PROCESS_NUM = 8
SHM_NAME = "human_eval_pile"
FUZZ_THRESHOLD = 50
STRIDE_PERCENT = 0.05

# Logging setup
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_best_match_for_test(test_tuple):
    """Find the best fuzzy match for a given test string from shared memory."""
    test_str, shm_name, threshold, stride_percent = test_tuple

    # Attach to the shared memory and decode it
    existing_shm = shared_memory.SharedMemory(name=shm_name)
    corpus_chunk = existing_shm.buf[:].tobytes().decode("utf-8")

    best_score = 0
    best_match = None
    stride = max(int(len(test_str) * stride_percent), 1)

    for i in range(0, len(corpus_chunk) - len(test_str), stride):
        score = fuzz.ratio(corpus_chunk[i : i + len(test_str)], test_str)
        if score > best_score:
            best_score = score
            best_match = corpus_chunk[i : i + len(test_str)]

    existing_shm.close()
    return (
        (test_str, best_match, best_score)
        if best_score >= threshold
        else (test_str, None, 0)
    )


def load_test_data(filename):
    """Load test data from the specified JSONL file."""
    with open(filename, "r") as f:
        return [json.loads(line)["canonical_solution"] for line in f.readlines()]


def load_corpus_data(corpus_files):
    """Load corpus data from the specified corpus files."""
    corpus_data = []
    for corpus_file in tqdm(corpus_files, desc="Loading corpus files"):
        with open(corpus_file, "r") as f:
            corpus_data.extend([json.loads(line)["text"] for line in f.readlines()])
    return corpus_data


def create_corpus_chunks(corpus_data, chunk_size):
    """Create fixed-size chunks from the corpus data."""
    chunks = []
    i = 0
    while i < len(corpus_data):
        str_builder = StringIO()
        while i < len(corpus_data) and str_builder.tell() < chunk_size:
            str_builder.write(corpus_data[i])
            i += 1
        chunks.append(str_builder.getvalue())
    return chunks


def main(input_path, result_dir):
    # Validate file extensions
    if not input_path.endswith(".jsonl"):
        raise ValueError("Input file must have a .jsonl extension")
    logger.info("Reading test file...")
    test_strs = load_test_data(input_path)

    logger.info("Reading training corpus...")
    corpus_data = load_corpus_data(CORPUS_FILES[:1])

    logger.info("Creating corpus chunks...")
    corpus_chunks = create_corpus_chunks(corpus_data, CHUNK_SIZE)
    logger.info(f"Created {len(corpus_chunks)} chunks")

    # Initialize results dictionary
    overall_results = {
        test_str: {"score": 0, "closest_solution": None} for test_str in test_strs
    }

    for chunk_str in tqdm(corpus_chunks[:1], desc="Processing chunks"):
        # Create shared memory and load the current chunk into it
        chunk_str_bytes = chunk_str.encode("utf-8")
        shm = shared_memory.SharedMemory(
            name=SHM_NAME, create=True, size=len(chunk_str_bytes)
        )
        shm.buf[: len(chunk_str_bytes)] = chunk_str_bytes

        # Prepare tasks for parallel processing
        task_args = [
            (test_str, SHM_NAME, FUZZ_THRESHOLD, STRIDE_PERCENT)
            for test_str in test_strs
        ]

        with Pool(PROCESS_NUM) as pool:
            chunk_results = pool.map(find_best_match_for_test, task_args)

        # Update overall results with the best matches from the current chunk
        for test_str, best_match, best_score in chunk_results:
            if best_score > overall_results[test_str]["score"]:
                overall_results[test_str]["score"] = best_score
                overall_results[test_str]["closest_solution"] = best_match

        # Clean up shared memory
        shm.close()
        shm.unlink()

    # Save the final results to a JSONL file
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, os.path.basename(input_path))
    with open(result_path, "w") as f:
        for test_str, data in overall_results.items():
            json.dump(
                {
                    "solution": test_str,
                    "closest_solution": data["closest_solution"],
                    "score": data["score"],
                },
                f,
            )
            f.write("\n")

    logger.info("Results have been saved to %s", result_path)


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

    args = parser.parse_args()

    main(args.input_path, args.result_dir)
