import os
import json
from tqdm import tqdm


from utils.constants import CORPUS_DIR, CORPUS_FILE_FORMAT, CORPUS_FILES_AMOUNT
from utils.data_loading import load_test_data, load_corpus_data
from utils.chunk_processing import (
    create_corpus_chunks,
    search_test_string_in_chunks,
    search_multiple_test_strings_in_chunks,
)
from utils.save_results import save_results
from utils.logger import logger
from utils.parse_arguments import parse_fuzzy_match_arguments

CORPUS_FILES = [
    CORPUS_DIR / CORPUS_FILE_FORMAT.format(part) for part in range(CORPUS_FILES_AMOUNT)
]


def main(
    input_path, result_dir, num_corpus_files, num_chunks_to_read, detailed_results
):
    logger.info("Loading test file...")
    test_data = load_test_data(input_path)
    is_single = isinstance(test_data, str)
    test_data = [test_data] if is_single else test_data
    logger.info(f"Test data has {len(test_data)} solutions")

    corpus_files = CORPUS_FILES[:num_corpus_files] if num_corpus_files else CORPUS_FILES
    result_file = os.path.basename(input_path)

    all_results = {
        test_text: {
            "solution": test_text,
            "score": 0,
            "closest_solution": None,
            "chunk_results": [] if detailed_results else None,
        }
        for test_text in test_data
    }
    start_idx = 0

    for corpus_file in tqdm(corpus_files, desc="Processing corpus files"):
        logger.info(f"Loading corpus data from {corpus_file}...")
        corpus_data = load_corpus_data(corpus_file)
        logger.info(f"Creating corpus chunks for {corpus_file}...")
        chunks = create_corpus_chunks(corpus_data, num_chunks_to_read, start_idx)
        start_idx += len(chunks)

        if is_single:
            results = search_test_string_in_chunks(
                test_data[0], chunks, detailed_results
            )
            all_results[test_data[0]].update(results)
            if results["score"] == 100:
                break
        else:
            results = search_multiple_test_strings_in_chunks(
                test_data, chunks, detailed_results
            )
            for test_text, result in results.items():
                all_results[test_text].update(result)

    save_results(
        all_results[test_data[0]] if is_single else all_results,
        result_dir,
        "FuzzyMatch" + result_file,
        is_single,
    )


if __name__ == "__main__":
    args = parse_fuzzy_match_arguments()

    main(
        args.input_path,
        args.result_dir,
        args.num_corpus_files,
        args.num_chunks_to_read,
        args.detailed_results,
    )
