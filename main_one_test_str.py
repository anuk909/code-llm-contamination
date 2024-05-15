import os
import json
import logging
from thefuzz import fuzz
from tqdm import tqdm
from io import StringIO
from concurrent.futures import ProcessPoolExecutor, as_completed

# Directory and file configurations
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

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def find_best_match_for_test(test_str, corpus_chunk, threshold, stride_percent):
    """
    Find the best fuzzy match for a given test string in a corpus chunk.
    :param test_str: The string to match against.
    :param corpus_chunk: The chunk of text from the corpus.
    :param threshold: Minimum score to accept a match.
    :param stride_percent: The stride percent to use for sliding window comparison.
    :return: Tuple of best match and its score or (None, 0) if no match exceeds threshold.
    """
    best_score = 0
    best_match = None
    stride = max(int(len(test_str) * stride_percent), 1)

    for i in range(0, len(corpus_chunk) - len(test_str), stride):
        score = fuzz.ratio(corpus_chunk[i : i + len(test_str)], test_str)
        if score > best_score:
            best_score = score
            best_match = corpus_chunk[i : i + len(test_str)]

    return (best_match, best_score) if best_score >= threshold else (None, 0)


def load_test_data(filename):
    """
    Load test data from a given JSONL file.
    :param filename: Path to the test data file.
    :return: Test string.
    """
    try:
        with open(filename, "r") as f:
            return json.loads(f.readline())["canonical_solution"]
    except Exception as e:
        logger.error(f"Error loading test data from {filename}: {e}")
        raise


def load_corpus_data(corpus_files):
    """
    Load corpus data from a list of corpus files.
    :param corpus_files: List of paths to corpus files.
    :return: List of text contents from corpus files.
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
    :param corpus_data: List of corpus data strings.
    :param chunk_size: Size of each chunk in characters.
    :param max_chunks: Maximum number of chunks to create (or None for no limit).
    :return: List of corpus chunks.
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


def search_test_string_in_corpus(test_str, corpus_chunks):
    """
    Search for the best matching string in the corpus chunks for a given test string.
    :param test_str: The test string to search for.
    :param corpus_chunks: List of corpus chunks.
    :return: Dictionary containing the test string, best match, and score.
    """
    best_match = None
    best_score = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                find_best_match_for_test,
                test_str,
                chunk,
                FUZZ_THRESHOLD,
                STRIDE_PERCENT,
            ): chunk
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


def save_results(result, result_dir, result_file):
    """
    Save the result as a JSONL file.
    :param result: The result dictionary to save.
    :param result_dir: The directory to save the result file.
    :param result_file: The name of the result file.
    """
    os.makedirs(result_dir, exist_ok=True)
    result_path = os.path.join(result_dir, result_file)
    try:
        with open(result_path, "w") as f:
            json.dump(result, f)
            f.write("\n")
        logger.info(f"Results have been saved successfully to {result_path}")
    except Exception as e:
        logger.error(f"Error saving results to {result_path}: {e}")
        raise


def main(input_path, result_dir, num_corpus_files, num_chunks_to_read):
    logger.info("Reading test file...")
    test_str = load_test_data(input_path)

    logger.info("Loading corpus data...")
    corpus_files = (
        CORPUS_FILES if num_corpus_files is None else CORPUS_FILES[:num_corpus_files]
    )
    corpus_data = load_corpus_data(corpus_files)

    logger.info("Creating corpus chunks...")
    corpus_chunks = create_corpus_chunks(corpus_data, CHUNK_SIZE, num_chunks_to_read)

    logger.info("Searching for solutions in corpus...")
    result = search_test_string_in_corpus(test_str, corpus_chunks)

    logger.info("Saving results...")
    save_results(result, result_dir, os.path.basename(input_path))


if __name__ == "__main__":
    import argparse

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

# Example usage - python main_one_test_str.py --input_path inputs/SmallHumanEval.jsonl --result_dir results --num_corpus_files 1 --num_chunks_to_read 88
