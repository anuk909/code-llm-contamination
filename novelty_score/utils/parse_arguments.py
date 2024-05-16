import argparse

from .constants import CORPUS_FILES_AMOUNT


def parse_fuzzy_match_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuzzy match test strings against corpus data."
    )
    parser.add_argument(
        "--input_path",
        required=True,
        type=str,
        help="Input path containing jsonl test file.",
    )
    parser.add_argument(
        "--result_dir",
        required=True,
        type=str,
        help="Directory path for saving the jsonl result file.",
    )
    parser.add_argument(
        "--num_corpus_files",
        type=int,
        default=None,
        help="Number of corpus files to process. Use None for no limit.",
    )
    parser.add_argument(
        "--max_chunks",
        type=int,
        default=None,
        help="Number of chunks to read for processing. Use None for no limit.",
    )
    parser.add_argument(
        "--detailed_results",
        action="store_true",
        help="Store chunk-level results in addition to the overall best match.",
    )

    args: argparse.Namespace = parser.parse_args()

    if not args.input_path.endswith(".jsonl"):
        raise ValueError("Input file must have a .jsonl extension")
    if args.num_corpus_files and args.num_corpus_files > CORPUS_FILES_AMOUNT:
        raise ValueError(
            f"Number of corpus files must be equal or less than {CORPUS_FILES_AMOUNT}"
        )
    return args


def parse_dolos_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Dolos Analysis.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Input path containing jsonl test file in the correct format.",
    )
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Directory path for saving the jsonl result file.",
    )
    args: argparse.Namespace = parser.parse_args()
    return args
